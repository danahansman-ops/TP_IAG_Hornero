"""
tools.py: tools de la etapa 1 (Marketing).

- buscar_guia_marca: RAG sobre guia_de_marca.pdf (índice FAISS guardado en disco).
- leer_historial:    lectura de contenidos_aprobados.csv (qué se le comunicó a cada segmento).
- guardar_pieza:     escritura de una pieza aprobada como fila nueva del mismo CSV.

Además hay funciones comunes (sin @tool) que usa el código del grafo para decidir qué recibe
cada agente: buscar_en_guia, guia_completa, ejemplos_de_canal, canales_disponibles.

Recordatorios (Clase 4):
- El docstring de cada tool es lo que el LLM lee para decidir cuándo usarla.
- Que cada tool reciba al menos un parámetro.
- Lo que devuelve una tool entra entero en el contexto del LLM: devolvemos resúmenes, no tablas enteras.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_core.tools import tool

from rag import IndiceRAG

# Carpeta con los datos del TP. Si mueven los archivos, ajusten esta ruta.
RUTA_DATOS = Path(__file__).parent / "datos"
RUTA_GUIA = RUTA_DATOS / "etapa1" / "guia_de_marca.pdf"
RUTA_CONTENIDOS = RUTA_DATOS / "etapa1" / "contenidos_aprobados.csv"
CARPETA_INDICES = Path(__file__).parent / "indices"

COLUMNAS = ["fecha", "canal", "segmento", "titulo", "texto"]
SEGMENTOS = ["Nuevos", "Recurrentes", "Premium", "Todos"]

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

# El índice se arma (o se levanta del disco) recién la primera vez que alguien busca
indice_guia = IndiceRAG(RUTA_GUIA, CARPETA_INDICES, "guia_marca")


# --------------------------------------------------------------------------
# Funciones comunes (las usa el código del grafo, no el LLM)
# --------------------------------------------------------------------------

def _leer_contenidos() -> pd.DataFrame:
    return pd.read_csv(RUTA_CONTENIDOS)


def _normalizar(texto: str) -> str:
    return str(texto).strip().lower()


def canales_disponibles() -> list[str]:
    """Canales que ya existen en el historial (para que el estratega use los mismos nombres)."""
    return sorted(_leer_contenidos()["canal"].dropna().unique().tolist())


def buscar_en_guia(consultas: list[str], k: int = 1) -> str:
    """Trae de la guía los fragmentos relevantes para varias consultas, sin repetir."""
    chunks = indice_guia.buscar_varias(consultas, k)
    return "\n\n---\n\n".join(chunks)


def guia_completa() -> str:
    """La guía entera, leída del PDF (es corta: unas 3 páginas). La usa el revisor."""
    return indice_guia.texto_completo()


def ejemplos_de_canal(canal: str, cantidad: int = 2) -> str:
    """Piezas ya aprobadas del mismo canal: sirven de ejemplo de piezas que cumplen la guía."""
    df = _leer_contenidos()
    df = df[df["canal"].map(_normalizar) == _normalizar(canal)]
    if df.empty:
        return "(no hay piezas aprobadas de este canal)"
    df = df.sort_values("fecha", ascending=False).head(cantidad)
    return "\n\n".join(f"TITULO: {f.titulo}\nTEXTO: {f.texto}" for f in df.itertuples())


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@tool
def buscar_guia_marca(consulta: str) -> str:
    """Busca en la guía de marca de Hornero las reglas relevantes para una consulta:
    voz y tono, palabras prohibidas, reglas por canal, legales obligatorios o cómo
    hablarle a un segmento. Ejemplo: consulta='reglas para WhatsApp'."""
    return "\n\n---\n\n".join(indice_guia.buscar(consulta, k=2))


@tool
def leer_historial(segmento: Literal["Nuevos", "Recurrentes", "Premium", "Todos"],
                   canal: str = "todos", cantidad: int = 6) -> str:
    """Devuelve las piezas que Marketing ya le mandó a un segmento (las más recientes primero),
    para no repetir mensajes. Incluye las piezas enviadas a 'Todos', porque también le llegaron
    a ese segmento. canal='todos' trae todos los canales; si no, filtra por ese canal
    (por ejemplo 'Instagram' o 'Email')."""
    df = _leer_contenidos()
    if segmento == "Todos":
        filtro = pd.Series(True, index=df.index)
    else:
        filtro = df["segmento"].map(_normalizar).isin([_normalizar(segmento), "todos"])
    if _normalizar(canal) != "todos":
        filtro &= df["canal"].map(_normalizar) == _normalizar(canal)
    df = df[filtro].sort_values("fecha", ascending=False).head(cantidad)

    if df.empty:
        return f"No hay piezas previas para el segmento {segmento} en el canal '{canal}'."
    lineas = [f"Últimas {len(df)} piezas para {segmento} (canal: {canal}):"]
    for f in df.itertuples():
        texto = str(f.texto)
        texto = texto[:250] + ("..." if len(texto) > 250 else "")
        lineas.append(f"- {f.fecha} | {f.canal} | {f.segmento} | {f.titulo} | {texto}")
    return "\n".join(lineas)


@tool
def guardar_pieza(canal: str, segmento: Literal["Nuevos", "Recurrentes", "Premium", "Todos"],
                  titulo: str, texto: str) -> str:
    """Guarda una pieza APROBADA como fila nueva de contenidos_aprobados.csv, con la fecha y hora actual.
    Usar solo con piezas que ya pasaron la revisión de marca."""
    if not (canal.strip() and titulo.strip() and texto.strip()):
        return "No se guardó: canal, título y texto no pueden estar vacíos."
    if segmento not in SEGMENTOS:
        return f"No se guardó: el segmento debe ser uno de {SEGMENTOS}."

    df = _leer_contenidos()
    repetida = (df["titulo"].map(_normalizar) == _normalizar(titulo)) & \
               (df["texto"].map(_normalizar) == _normalizar(texto))
    if repetida.any():
        return "No se guardó: esa pieza ya está en el historial."

    # Mismo formato de fecha que las filas originales: "2026-10-05 10:30"
    fila = pd.DataFrame([{"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "canal": canal.strip(),
                          "segmento": segmento, "titulo": titulo.strip(), "texto": texto.strip()}])
    pd.concat([df, fila[COLUMNAS]], ignore_index=True).to_csv(RUTA_CONTENIDOS, index=False)
    return f"Pieza guardada en el historial ({len(df) + 1} piezas en total)."


@tool
def fecha_de_hoy(formato: Literal["numerico", "texto"]) -> str:
    """Devuelve la fecha de hoy. formato='numerico' devuelve AAAA-MM-DD;
    formato='texto' devuelve, por ejemplo, 'sábado 7 de noviembre de 2026'."""
    hoy = date.today()
    if formato == "texto":
        return f"{DIAS[hoy.weekday()]} {hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}"
    return hoy.isoformat()