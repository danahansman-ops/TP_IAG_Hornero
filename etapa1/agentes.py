"""
agentes.py: etapa 1, pipeline secuencial de Marketing con LangGraph.

    pedido ─► ESTRATEGA ─► REDACTOR ─► REVISOR ─► GUARDAR ─► respuesta
                 │            ▲           │
                 │            └─ rechaza ─┘   (como máximo MAX_BORRADORES borradores)
                 └─ fuera de alcance ─► fin

- Estratega (LLM + tool leer_historial): entiende el pedido y arma un brief. Es el único
  que ve el historial, para no repetir lo que ya se le mandó al segmento.
- Redactor (LLM): escribe la pieza. Recibe solo el brief, las secciones de la guía que trae
  el RAG y dos piezas aprobadas del mismo canal como ejemplo de formato.
- Revisor de marca (LLM): lee la guía de marca COMPLETA y controla la pieza contra todas sus
  reglas. Recibe solo la pieza, el pedido original y las medidas de la pieza.
- Guardar (código): si la pieza se aprobó, la agrega al CSV (siempre, salvo que el pedido diga
  explícitamente que no; ver GUARDAR_SIEMPRE).

Todas las reglas salen del PDF de la guía: el código no tiene reglas de marca escritas a mano.
Qué recibe cada agente lo decide el código (las funciones nodo_*), no un LLM. Lo que el código
puede verificar no se deja librado al LLM: el canal y el segmento se validan contra el historial,
el veredicto solo aprueba si dice exactamente APROBADA, y las medidas (largo, palabras, emojis)
se calculan con código.

main.py solo usa responder(). La etapa 2 reutiliza ejecutar_marketing().
"""

import operator
import os
import re
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from tools import (SEGMENTOS, buscar_en_guia, canales_disponibles, ejemplos_de_canal,
                   guardar_pieza, guia_completa, leer_historial, normalizar_canal)

load_dotenv()

MODELO = "openai/gpt-oss-20b"
# reasoning_effort="low": con el valor por defecto, gpt-oss-20b a veces razona hasta agotar los
# tokens de salida y devuelve una respuesta vacía. Además gasta muchos menos tokens.
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=MODELO, temperature=0.0, max_retries=5,
               reasoning_effort="low")

# Ciclo de corrección: si el revisor rechaza, el redactor reescribe UNA vez con las observaciones.
# Es lo que hace un equipo de Marketing real (escribe → revisa → corrige) y evita descartar una
# pieza por un detalle corregible. Se limita a 2 borradores para acotar tiempo y llamadas a Groq.
MAX_BORRADORES = 2        # 1 borrador + 1 corrección como máximo (cada vuelta suma 2 llamadas al LLM)
# True: guarda toda pieza aprobada, porque la consigna pide entregar el CSV con las piezas de
# TODAS las consultas de prueba. Un pedido que dice explícitamente "no lo guardes" gana igual.
# False: guarda solo si el pedido lo pide ("guardalo").
GUARDAR_SIEMPRE = True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def preguntar(agente, texto: str) -> str:
    resultado = agente.invoke({"messages": [{"role": "user", "content": texto}]})
    return resultado["messages"][-1].content


def tools_llamadas(resultado) -> str:
    return "\n".join(f"→ Tool: {tc['name']}({tc['args']})"
                     for msg in resultado["messages"] for tc in (getattr(msg, "tool_calls", None) or []))


def limpiar(texto: str) -> str:
    # Saca el markdown en negrita que a veces agrega el modelo ("**CANAL:**")
    return re.sub(r"\*\*", "", texto or "").strip()


def extraer_campo(texto: str, campo: str, siguientes: list[str] | None = None) -> str:
    """Extrae el valor de 'CAMPO: valor'. Si se pasan los campos siguientes, toma todo el
    bloque (varias líneas) hasta el próximo campo; si no, solo esa línea."""
    texto = limpiar(texto)
    if siguientes:
        fin = "|".join(re.escape(s) for s in siguientes)
        patron = rf"^{campo}\s*:\s*(.*?)(?=\n\s*(?:{fin})\s*:|\Z)"
    else:
        patron = rf"^{campo}\s*:\s*([^\n]*)"
    m = re.search(patron, texto, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ""


EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿]")
ROTULOS = ["Asunto", "Título", "Texto"]
ROTULO = re.compile(rf"^\s*(?:{'|'.join(ROTULOS)})\s*:\s*", flags=re.IGNORECASE | re.MULTILINE)


def palabras_de(texto: str) -> int:
    return len(re.findall(r"\w+", texto))


def cuerpo_de_email(texto: str) -> tuple[str | None, str]:
    """Separa el cuerpo de un email: las líneas entre el saludo ("Hola, {nombre}:") y el botón
    ("[Botón: ...]"). Quedan afuera el asunto, el saludo, el botón y los legales del pie.
    Devuelve (cuerpo, "") o (None, motivo) si no encuentra el saludo o el botón."""
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    lineas = [l for l in lineas if not re.match(r"asunto\s*:", l, flags=re.IGNORECASE)]
    saludo = next((i for i, l in enumerate(lineas) if re.match(r"hola\b", l, flags=re.IGNORECASE)), None)
    boton = next((i for i, l in enumerate(lineas) if re.fullmatch(r"\[.*\]", l)), None)
    if saludo is None:
        return None, "no se encontró la línea de saludo (\"Hola, ...\")"
    if boton is None or boton <= saludo:
        return None, "no se encontró una línea de botón entre corchetes después del saludo"
    return "\n".join(lineas[saludo + 1:boton]), ""


def medir_pieza(texto: str, canal: str = "") -> str:
    """Medidas objetivas de la pieza, calculadas con código para que el revisor las compare
    con las reglas de la guía. Los LLM cuentan mal caracteres y emojis; acá no hay reglas,
    solo conteos: los límites los pone la guía. Cada medida dice exactamente qué cubre."""
    sin_rotulos = ROTULO.sub("", texto)
    hashtags = len(re.findall(r"#\w+", texto))
    medidas = [f"- Caracteres de toda la pieza (sin rótulos como 'Título:'): {len(sin_rotulos)}",
               f"- Palabras de toda la pieza (sin rótulos): {palabras_de(sin_rotulos)}",
               f"- Hashtags: {hashtags}",
               f"- Emojis en toda la pieza: {len(EMOJI.findall(texto))}"]
    # Si la pieza tiene líneas con rótulo (Asunto:, Título:, Texto:), medimos cada una sin el rótulo
    for rotulo in ROTULOS:
        m = re.search(rf"^\s*{rotulo}\s*:\s*([^\n]*)", texto, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            linea = m.group(1).strip()
            medidas.append(f"- Línea '{rotulo}' (sin el rótulo): {len(linea)} caracteres, "
                           f"{len(EMOJI.findall(linea))} emojis")
    # Email: la guía pone el tope de palabras al cuerpo, no a toda la pieza
    if (normalizar_canal(canal) or "") == "Email":
        cuerpo, motivo = cuerpo_de_email(texto)
        if cuerpo is None:
            medidas.append(f"- Cuerpo del email: no se pudo medir ({motivo}); contalo vos")
        else:
            medidas.append(f"- Cuerpo del email (entre el saludo y el botón; sin asunto, saludo, "
                           f"botón ni legales del pie): {palabras_de(cuerpo)} palabras")
    return "\n".join(medidas)


# --------------------------------------------------------------------------
# System prompts (rol, scope y tono de cada agente)
# --------------------------------------------------------------------------

REGLA_NO_INVENTAR = """NO INVENTAR DATOS:
- Nunca inventes datos que no estén en el pedido: precios, montos, porcentajes o códigos de
  descuento, fechas de vigencia, plazos de entrega, stock, cuotas, nombres de productos o colecciones.
- Tampoco inventes características del producto (materiales, terminaciones, medidas, garantías,
  "artesanal", "premium") ni beneficios que el pedido no menciona. Contá para qué sirve, no cómo está hecho.
- Si la pieza necesita un dato que el pedido no da (por ejemplo la vigencia de un descuento),
  dejá un marcador entre corchetes para que Marketing lo complete: [DD/MM], [X%], [CÓDIGO]."""

SYSTEM_ESTRATEGA = f"""Sos estratega de contenidos del equipo de Marketing de Hornero, un e-commerce
argentino de artículos para el hogar y electro que vende por web, app y marketplaces.

ROL: a partir de un pedido en lenguaje natural, definís QUÉ comunicar y CÓMO. No escribís la pieza.

TONO (de tu respuesta): profesional, concreto y breve. El brief es un documento interno para el
redactor: sin adornos ni frases publicitarias.

SCOPE:
- Solo pedidos de piezas de comunicación para clientes de Hornero (posts, emails, mensajes de
  WhatsApp, push, LinkedIn). Si el pedido es cualquier otra cosa, respondé únicamente: FUERA DE ALCANCE
- Antes de armar el brief, usá SIEMPRE la tool leer_historial con el segmento destinatario para
  ver qué se le comunicó hace poco, y proponé un ángulo que no repita esos mensajes.
- Usá los fragmentos de la guía de marca que te pasamos para elegir el tono según el segmento
  y la situación.

{REGLA_NO_INVENTAR}

FORMATO DE RESPUESTA (exactamente estos campos, en este orden, sin markdown):
CANAL: <uno de los canales que te indico en el pedido>
SEGMENTO: <Nuevos, Recurrentes, Premium o Todos>
SITUACIÓN: <promoción, lanzamiento, recuperación de clientes o contenido>
OBJETIVO: <qué queremos lograr con la pieza, en una oración>
DATOS DEL PEDIDO: <solo los datos concretos que dio el pedido (categoría, descuento, etc.) o "ninguno">
YA COMUNICADO: <mensajes recientes a este segmento que hay que evitar repetir>
ÁNGULO: <la idea central de la pieza, distinta de lo ya comunicado>
TONO: <cómo hablarle a este segmento en esta situación, según la guía>

Si el pedido no dice el segmento, elegí el más adecuado; si ninguno aplica, usá Todos.
Respondé en español rioplatense."""

SYSTEM_REDACTOR = f"""Sos redactor/a publicitario/a de Hornero, un e-commerce argentino de artículos para
el hogar y electro.

ROL: escribís UNA pieza de comunicación siguiendo el brief y las reglas de la guía de marca que
te pasamos. Las piezas son solo texto.

SCOPE:
- Solo escribís la pieza que pide el brief, para el canal y el segmento que indica. No cambiás
  el canal, el segmento ni el ángulo, y no agregás piezas extra ni comentarios fuera del formato.
- Tu única fuente de reglas son los fragmentos de la guía; tu única fuente de datos, el brief.

TONO: el que indica la guía para el segmento y la situación del brief, en español rioplatense
con voseo: cálido, claro y práctico, como un vecino que sabe de casas.

REGLAS:
- Seguí todas las reglas de los fragmentos de la guía: voz y tono, reglas del canal (largo,
  hashtags, emojis, llamado a la acción), palabras prohibidas y legales obligatorios.
- Copiá el FORMATO de las piezas aprobadas del mismo canal (por ejemplo, si los emails empiezan
  con "Asunto:" o los push tienen líneas "Título:" y "Texto:"), pero no copies sus mensajes.
- Si recibís observaciones de la revisión, corregí TODAS sin cambiar lo que estaba bien.

{REGLA_NO_INVENTAR}

FORMATO DE RESPUESTA (exactamente estos tres campos, sin markdown):
TITULO: <título interno corto y descriptivo de la pieza, como los del historial>
TEXTO: <la pieza completa tal como se publica, con el mismo formato que las piezas aprobadas del canal>
IMAGEN SUGERIDA: <una sola línea describiendo la imagen que podría acompañarla>"""

SYSTEM_REVISOR = """Sos revisor/a de marca de Hornero. Controlás que una pieza cumpla la guía de marca
antes de publicarse.

ROL: revisás, no reescribís. Recibís la guía de marca completa, el pedido original, la pieza y sus
medidas (caracteres, palabras, hashtags, emojis), que fueron calculadas con exactitud: usalas en
lugar de contar vos. Fijate qué parte de la pieza cubre cada medida antes de compararla con la guía.

SCOPE:
- Solo controlás la pieza contra la guía de marca y el pedido original. No evaluás la estrategia
  ni el ángulo, y no proponés una versión reescrita de la pieza.
- Solo marcás incumplimientos que puedas justificar citando la guía o el pedido.

TONO: objetivo, preciso y breve. Cada observación nombra la regla incumplida y qué cambiar.

QUÉ CONTROLÁS (todo según la guía):
1. Reglas del canal: largo, hashtags, emojis, llamado a la acción y reglas generales.
2. Palabras y frases prohibidas.
3. Plazos de entrega: no se promete ninguno.
4. Legales obligatorios si hay descuento, cuotas o envío sin cargo, en el lugar que indica la guía.
5. Voz y tono adecuados al segmento y a la situación.
6. Datos inventados: cualquier precio, descuento, fecha, producto o característica del producto
   que NO esté en el pedido es un error. Los marcadores entre corchetes como [DD/MM] o [X%] están
   permitidos: son la forma correcta de dejar datos pendientes.

Si la pieza cumple, aprobala aunque se pudiera escribir de otra manera.

FORMATO DE RESPUESTA (exactamente estos campos, sin markdown):
VEREDICTO: <solo la palabra APROBADA o la palabra RECHAZADA>
OBSERVACIONES: <si está rechazada, lista de incumplimientos concretos, la regla de la guía que
incumplen y cómo corregirlos; si está aprobada, "ninguna">"""

# Tres agentes con create_agent, cada uno con sus tools (mínimo privilegio):
# solo el estratega lee el historial; nadie escribe en el CSV (lo hace el nodo de código).
agente_estratega = create_agent(model=llm, tools=[leer_historial], system_prompt=SYSTEM_ESTRATEGA)
agente_redactor = create_agent(model=llm, tools=[], system_prompt=SYSTEM_REDACTOR)
agente_revisor = create_agent(model=llm, tools=[], system_prompt=SYSTEM_REVISOR)


# --------------------------------------------------------------------------
# Estado del grafo
# --------------------------------------------------------------------------

class EstadoMarketing(TypedDict, total=False):
    pedido: str
    guardar: bool | None   # True: el pedido pide guardar; False: pide NO guardar; None: no dice
    # Estratega
    brief: str
    canal: str
    segmento: str
    fuera_de_alcance: bool
    sin_canal: bool        # el canal del brief y del pedido no es ninguno de los del historial
    # Redactor
    titulo: str
    texto: str
    imagen: str
    borradores: int
    # Revisor
    aprobada: bool
    observaciones: str
    # Guardado
    resultado_guardado: str
    # Pasos para mostrar en la app (cada nodo agrega los suyos)
    pasos: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------
# Nodos
# --------------------------------------------------------------------------

def nodo_estratega(estado: EstadoMarketing) -> dict:
    # RAG: secciones de la guía para elegir el tono según segmento y situación
    guia = buscar_en_guia(["segmentos de clientes y cómo hablarles", "voz y tono según la situación"])
    entrada = (f"PEDIDO: {estado['pedido']}\n\n"
               f"CANALES DE HORNERO (usá estos nombres): {', '.join(canales_disponibles())}\n\n"
               f"FRAGMENTOS DE LA GUÍA DE MARCA:\n{guia}")
    resultado = agente_estratega.invoke({"messages": [{"role": "user", "content": entrada}]})
    brief = limpiar(resultado["messages"][-1].content)

    paso = {"titulo": "1. Estratega (brief)", "contenido": f"{tools_llamadas(resultado)}\n\n{brief}".strip()}
    if "FUERA DE ALCANCE" in brief.upper():
        return {"fuera_de_alcance": True, "brief": brief, "pasos": [paso]}

    segmento = (extraer_campo(brief, "SEGMENTO").split() or ["Todos"])[0].strip(".,")
    segmento = next((s for s in SEGMENTOS if s.lower() == segmento.lower()), "Todos")
    # El canal se valida con código, igual que el segmento: se lleva al nombre exacto del
    # historial ("Instagram (post)" → "Instagram"). Si el brief no trae uno válido, se busca
    # en el pedido; si tampoco, se corta el flujo en vez de seguir con un canal desconocido.
    canal = normalizar_canal(extraer_campo(brief, "CANAL")) or normalizar_canal(estado["pedido"])
    if canal is None:
        return {"fuera_de_alcance": False, "sin_canal": True, "brief": brief, "pasos": [paso]}
    return {
        "fuera_de_alcance": False,
        "sin_canal": False,
        "brief": brief,
        "canal": canal,
        "segmento": segmento,
        "borradores": 0,
        "pasos": [paso],
    }


def nodo_redactor(estado: EstadoMarketing) -> dict:
    canal, segmento = estado["canal"], estado["segmento"]
    # RAG: el código decide qué secciones de la guía necesita el redactor
    guia = buscar_en_guia(["voz y tono", f"reglas por canal {canal}", f"cómo hablarle al segmento {segmento}",
                           "palabras y frases prohibidas", "promociones y legales obligatorios",
                           "plazos de entrega"])
    entrada = (f"BRIEF:\n{estado['brief']}\n\n"
               f"FRAGMENTOS DE LA GUÍA DE MARCA:\n{guia}\n\n"
               f"PIEZAS APROBADAS DEL CANAL {canal.upper()} (ejemplos de formato y estilo):\n{ejemplos_de_canal(canal)}")
    if estado.get("observaciones") and not estado.get("aprobada", True):
        entrada += (f"\n\nTU BORRADOR ANTERIOR:\nTITULO: {estado['titulo']}\nTEXTO: {estado['texto']}\n\n"
                    f"OBSERVACIONES DE LA REVISIÓN (corregí todas):\n{estado['observaciones']}")

    pieza = limpiar(preguntar(agente_redactor, entrada))
    n = estado.get("borradores", 0) + 1
    return {
        "titulo": extraer_campo(pieza, "TITULO", ["TEXTO"]),
        "texto": extraer_campo(pieza, "TEXTO", ["IMAGEN SUGERIDA"]),
        "imagen": extraer_campo(pieza, "IMAGEN SUGERIDA"),
        "borradores": n,
        "pasos": [{"titulo": f"2. Redactor (borrador {n})", "contenido": pieza}],
    }


def nodo_revisor(estado: EstadoMarketing) -> dict:
    # El revisor lee la guía COMPLETA (es corta) para controlar todas las reglas, y recibe
    # solo la pieza y el pedido: no ve el historial ni el brief.
    medidas = medir_pieza(estado["texto"], estado["canal"])
    entrada = (f"GUÍA DE MARCA COMPLETA:\n{guia_completa()}\n\n"
               f"========\n"
               f"CANAL: {estado['canal']}\nSEGMENTO: {estado['segmento']}\n"
               f"PEDIDO ORIGINAL: {estado['pedido']}\n\n"
               f"PIEZA A REVISAR:\nTITULO: {estado['titulo']}\nTEXTO:\n{estado['texto']}\n\n"
               f"MEDIDAS DE LA PIEZA (calculadas con exactitud):\n{medidas}")
    revision = limpiar(preguntar(agente_revisor, entrada))
    aprobada, observaciones = leer_revision(revision)
    return {
        "aprobada": aprobada,
        "observaciones": observaciones,
        "pasos": [{"titulo": f"3. Revisor de marca (borrador {estado['borradores']})",
                   "contenido": f"{medidas}\n\n{revision}"}],
    }


def leer_revision(revision: str) -> tuple[bool, str]:
    """Interpreta la respuesta del revisor. Falla cerrado: solo se aprueba si el veredicto es
    exactamente APROBADA ("NO APROBADA", "APROBADA con cambios" o un veredicto vacío se
    toman como rechazo). Si rechaza sin observaciones, arma una para que el redactor sepa
    que tiene que revisar la pieza y el usuario no vea observaciones vacías."""
    veredicto = extraer_campo(revision, "VEREDICTO").upper().strip(" .!¡:")
    if veredicto == "APROBADA":
        return True, ""
    observaciones = extraer_campo(revision, "OBSERVACIONES", ["VEREDICTO"])
    if observaciones.strip().lower().strip(" .") in ("", "ninguna"):
        if veredicto == "RECHAZADA":
            observaciones = "El revisor rechazó la pieza sin detallar observaciones."
        else:
            observaciones = (f"El revisor no devolvió un veredicto válido (APROBADA o RECHAZADA): "
                             f"'{veredicto or 'vacío'}'.")
        observaciones += " Revisá la pieza contra todas las reglas de la guía y corregí lo que no cumpla."
    return False, observaciones


def nodo_guardar(estado: EstadoMarketing) -> dict:
    pedido_guardar = estado.get("guardar")   # True / False / None (ver pide_guardar)
    if not estado.get("aprobada"):
        msg = "No se guardó: la pieza no pasó la revisión de marca."
    elif pedido_guardar is False:
        msg = "No se guardó (el pedido pidió no guardarla)."
    elif not (pedido_guardar or GUARDAR_SIEMPRE):
        msg = "No se guardó (el pedido no lo pidió)."
    else:
        msg = guardar_pieza.invoke({"canal": estado["canal"], "segmento": estado["segmento"],
                                    "titulo": estado["titulo"], "texto": estado["texto"]})
    return {"resultado_guardado": msg, "pasos": [{"titulo": "4. Guardado", "contenido": msg}]}


# --------------------------------------------------------------------------
# Aristas condicionales
# --------------------------------------------------------------------------

def despues_del_estratega(estado: EstadoMarketing) -> str:
    return "fin" if estado.get("fuera_de_alcance") or estado.get("sin_canal") else "redactor"


def despues_del_revisor(estado: EstadoMarketing) -> str:
    if estado["aprobada"] or estado["borradores"] >= MAX_BORRADORES:
        return "guardar"
    return "redactor"   # vuelve a escribir con las observaciones


grafo = StateGraph(EstadoMarketing)
grafo.add_node("estratega", nodo_estratega)
grafo.add_node("redactor", nodo_redactor)
grafo.add_node("revisor", nodo_revisor)
grafo.add_node("guardar", nodo_guardar)
grafo.add_edge(START, "estratega")
grafo.add_conditional_edges("estratega", despues_del_estratega, {"redactor": "redactor", "fin": END})
grafo.add_edge("redactor", "revisor")
grafo.add_conditional_edges("revisor", despues_del_revisor, {"redactor": "redactor", "guardar": "guardar"})
grafo.add_edge("guardar", END)
flujo_marketing = grafo.compile()


# --------------------------------------------------------------------------
# Puntos de entrada
# --------------------------------------------------------------------------

# Pedir guardar la pieza: el verbo tiene que tener a la pieza como objeto. Con "guard" a secas,
# "guardarropas" o "guardá tus cosas" contaban como pedido de guardado.
_GUARDAR_PIEZA = (
    r"gu[aá]rd(?:[aá]|ar|e|en)l[oa]s?\b"                            # guardalo, guárdala, guardarlo
    r"|l[oa]s?\s+gu[aá]rd(?:[aá]s|es|en|ar|amos)\b"                 # lo guardás, la guardes
    r"|gu[aá]rd\w*\s+(?:\w+\s+){0,3}?en\s+el\s+(?:historial|csv)\b"  # guardá la pieza en el historial
)
GUARDAR_SI = re.compile(rf"\b(?:{_GUARDAR_PIEZA})")
# "no lo guardes", "sin guardarla", "no hace falta guardarlo" (pero no "tips para no guardar ropa")
GUARDAR_NO = re.compile(rf"\b(?:no|sin)\s+(?:(?:hace\s+falta|es\s+necesario|quiero\s+que)\s+)?"
                        rf"(?:{_GUARDAR_PIEZA})")


def pide_guardar(pedido: str) -> bool | None:
    """True si el pedido pide guardar la pieza, False si pide explícitamente NO guardarla,
    None si no dice nada (ahí decide GUARDAR_SIEMPRE)."""
    texto = pedido.lower()
    if GUARDAR_NO.search(texto):
        return False
    return True if GUARDAR_SI.search(texto) else None


def formatear_respuesta(estado: EstadoMarketing) -> str:
    if estado.get("fuera_de_alcance"):
        return ("Solo puedo ayudarte a crear piezas de comunicación para clientes de Hornero "
                "(posts de Instagram o LinkedIn, emails, mensajes de WhatsApp o push).")
    if estado.get("sin_canal"):
        return (f"No pude identificar el canal de la pieza. Decime en cuál de estos canales la "
                f"querés: {', '.join(canales_disponibles())}.")
    texto_md = estado["texto"].replace("\n", "  \n")   # en markdown, "  \n" fuerza el salto de línea
    partes = [f"**Canal:** {estado['canal']} · **Segmento:** {estado['segmento']}",
              f"**Título interno:** {estado['titulo']}",
              texto_md,
              f"*Imagen sugerida:* {estado['imagen']}"]
    if estado.get("aprobada"):
        partes.append(f"✅ Aprobada por la revisión de marca ({estado['borradores']} borrador/es).")
    else:
        obs = estado.get("observaciones", "").replace("\n", "  \n")
        partes.append(f"⚠️ No pasó la revisión de marca después de {estado['borradores']} borradores. "
                      f"Observaciones pendientes:  \n{obs}")
    partes.append(estado.get("resultado_guardado", ""))
    return "\n\n".join(partes)


def ejecutar_marketing(pedido: str) -> dict:
    """Corre el flujo completo. Devuelve {"respuesta", "detalle", "estado"}.
    La etapa 2 la reutiliza para integrar Marketing como un área más."""
    estado = flujo_marketing.invoke({"pedido": pedido, "guardar": pide_guardar(pedido), "pasos": []})
    return {"respuesta": formatear_respuesta(estado), "detalle": estado.get("pasos", []), "estado": estado}


def responder(pregunta: str) -> dict:
    """Punto de entrada que usa main.py: {"respuesta": str, "detalle": [{"titulo", "contenido"}]}."""
    inicio = time.perf_counter()
    try:
        resultado = ejecutar_marketing(pregunta)
        respuesta, detalle = resultado["respuesta"], resultado["detalle"]
    except Exception as e:  # que la app muestre el error en vez de romperse
        respuesta, detalle = f"Ocurrió un error al procesar el pedido: {e}", []
    segundos = time.perf_counter() - inicio
    detalle.append({"titulo": "⏱ Tiempo de respuesta", "contenido": f"{segundos:.1f} segundos"})
    return {"respuesta": respuesta, "detalle": detalle}