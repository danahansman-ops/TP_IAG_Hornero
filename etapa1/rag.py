"""
rag.py: RAG sobre un PDF con embeddings + FAISS (Clase 1), con el índice guardado en disco.

Es genérico a propósito: en la etapa 1 lo usamos para la guía de marca y en la etapa 2
lo reutilizamos tal cual para el manual de RRHH.

Cómo se lee el PDF:
- Las tablas se convierten a texto fila por fila ("Canal: WhatsApp | Largo máximo: 300 caracteres
  | ..."). Con extract_text() común las columnas se mezclan y las reglas quedan ilegibles.
- Se descartan los pies de página.

Cómo se parte el documento:
- Si el PDF tiene secciones numeradas ("3. Palabras y frases prohibidas"), cada sección es un
  chunk. Así una búsqueda trae la regla completa y no media sección cortada.
- Las secciones muy largas se subdividen, repitiendo el título de la sección en cada parte.
- Si el PDF no tiene secciones numeradas, se usa la ventana con solapamiento de la Clase 1.

Cómo funciona el cache:
- La primera vez lee el PDF, lo parte en chunks, calcula los embeddings y guarda dos archivos:
  <nombre>.faiss (los vectores) y <nombre>_chunks.json (los textos).
- Las siguientes veces levanta esos archivos directamente (mucho más rápido).
- Se vuelve a armar solo si el PDF es más nuevo que el índice o si cambió la forma de partirlo
  (VERSION_CHUNKING).
"""

import json
import re
from pathlib import Path

import numpy as np
import pdfplumber

# Mismo modelo que en la Clase 1: multilingüe, liviano, vectores de 384 dimensiones
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
# Si cambian la forma de partir el texto, suban este número: el índice se rearma solo
VERSION_CHUNKING = 3

_modelo = None  # se carga una sola vez, la primera vez que se necesita


def modelo_embeddings():
    global _modelo
    if _modelo is None:
        from sentence_transformers import SentenceTransformer   # import pesado: solo si hace falta
        _modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    return _modelo


def _tabla_a_texto(filas: list[list]) -> str:
    """Convierte una tabla en texto legible, una línea por fila:
    '- Canal: Instagram (post) | Largo máximo: 2.200 caracteres; ideal 150-300 | Hashtags: ...'
    Con extract_text() las columnas de la tabla se mezclan y el LLM no puede leerlas."""
    def celda(c):
        return " ".join(str(c or "").split())   # une los saltos de línea dentro de la celda
    encabezado = [celda(c) for c in filas[0]]
    lineas = []
    for fila in filas[1:]:
        pares = [f"{h}: {celda(v)}" for h, v in zip(encabezado, fila) if celda(v)]
        lineas.append("- " + " | ".join(pares))
    return "\n".join(lineas)


def _texto_de_pagina(pagina) -> str:
    """Texto de la página respetando las tablas: el texto normal se extrae como siempre y
    cada tabla se reemplaza por su versión fila por fila, en el lugar donde aparece."""
    tablas = sorted(pagina.find_tables(), key=lambda t: t.bbox[1])
    partes, y = [], 0
    for tabla in tablas:
        _, arriba, _, abajo = tabla.bbox
        if arriba > y:
            partes.append(pagina.crop((0, y, pagina.width, arriba)).extract_text() or "")
        partes.append(_tabla_a_texto(tabla.extract()))
        y = abajo
    partes.append(pagina.crop((0, y, pagina.width, pagina.height)).extract_text() or "")
    return "\n".join(p for p in partes if p.strip())


def leer_pdf(ruta_pdf: Path) -> str:
    with pdfplumber.open(ruta_pdf) as pdf:
        texto = "\n".join(_texto_de_pagina(p) for p in pdf.pages)
    # Sacamos los pies de página ("... Página 2"): no aportan y ensucian los chunks
    lineas = [l for l in texto.split("\n") if not re.search(r"P[aá]gina \d+\s*$", l)]
    return "\n".join(lineas)


def dividir_en_chunks(texto: str, tam_chunk: int = 600, solapamiento: int = 100) -> list[str]:
    """Ventana de caracteres con solapamiento (Clase 1), cortando en saltos de línea."""
    chunks, inicio = [], 0
    while inicio < len(texto):
        fin = min(inicio + tam_chunk, len(texto))
        if fin < len(texto):
            corte = texto.rfind("\n", inicio + tam_chunk // 2, fin)
            if corte != -1:
                fin = corte
        chunk = texto[inicio:fin].strip()
        if chunk:
            chunks.append(chunk)
        if fin >= len(texto):
            break
        # El solapamiento arranca al principio de una línea, para no empezar con media oración
        nuevo_inicio = texto.find("\n", fin - solapamiento, fin)
        inicio = nuevo_inicio + 1 if nuevo_inicio != -1 else max(fin - solapamiento, inicio + 1)
    return chunks


def dividir_por_secciones(texto: str, tam_max: int = 1800) -> list[str]:
    """Un chunk por sección numerada ("1. ...", "2. ..."). Si no encuentra al menos 3
    secciones, vuelve a la ventana con solapamiento."""
    titulo_seccion = re.compile(r"^\s*\d{1,2}\.\s+[A-ZÁÉÍÓÚÑ¿]")
    secciones, actual = [], []
    for linea in texto.split("\n"):
        if titulo_seccion.match(linea) and actual:
            secciones.append("\n".join(actual).strip())
            actual = []
        actual.append(linea)
    if actual:
        secciones.append("\n".join(actual).strip())

    if len(secciones) < 3:
        return dividir_en_chunks(texto)

    chunks = []
    for seccion in secciones:
        if not seccion:
            continue
        if len(seccion) <= tam_max:
            chunks.append(seccion)
            continue
        titulo = seccion.split("\n")[0]
        for i, parte in enumerate(dividir_en_chunks(seccion, tam_chunk=1200, solapamiento=150)):
            chunks.append(parte if i == 0 else f"{titulo} (continuación)\n{parte}")
    return chunks


def _embeber(textos: list[str]) -> np.ndarray:
    vectores = modelo_embeddings().encode(textos)
    # Normalizamos: con IndexFlatIP, producto interno = similitud coseno
    vectores = vectores / np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores.astype(np.float32)


class IndiceRAG:
    def __init__(self, ruta_pdf: Path, carpeta_indice: Path, nombre: str):
        self.ruta_pdf = Path(ruta_pdf)
        carpeta_indice = Path(carpeta_indice)
        self.ruta_faiss = carpeta_indice / f"{nombre}.faiss"
        self.ruta_chunks = carpeta_indice / f"{nombre}_chunks.json"
        self._indice = None
        self._chunks = None

    def _leer_guardado(self):
        """Devuelve los chunks guardados si el índice está al día; si no, None."""
        if not (self.ruta_faiss.exists() and self.ruta_chunks.exists()):
            return None
        if self.ruta_faiss.stat().st_mtime < self.ruta_pdf.stat().st_mtime:
            return None   # el PDF cambió después de armar el índice
        with open(self.ruta_chunks, encoding="utf-8") as f:
            guardado = json.load(f)
        if not isinstance(guardado, dict) or guardado.get("version") != VERSION_CHUNKING:
            return None   # se armó con otra forma de partir el texto
        return guardado["chunks"]

    def _cargar_o_construir(self):
        import faiss
        if self._indice is not None:
            return
        chunks = self._leer_guardado()
        if chunks is not None:
            self._indice = faiss.read_index(str(self.ruta_faiss))
            self._chunks = chunks
            return
        # Construir desde cero y guardar
        self._chunks = dividir_por_secciones(leer_pdf(self.ruta_pdf))
        vectores = _embeber(self._chunks)
        self._indice = faiss.IndexFlatIP(vectores.shape[1])
        self._indice.add(vectores)
        self.ruta_faiss.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._indice, str(self.ruta_faiss))
        with open(self.ruta_chunks, "w", encoding="utf-8") as f:
            json.dump({"version": VERSION_CHUNKING, "chunks": self._chunks}, f, ensure_ascii=False, indent=1)

    def texto_completo(self) -> str:
        """Todo el documento (todos los chunks en orden). Para documentos cortos, como la guía
        de marca, sirve cuando un agente tiene que controlar TODAS las reglas."""
        self._cargar_o_construir()
        return "\n\n".join(self._chunks)

    def buscar(self, consulta: str, k: int = 2) -> list[str]:
        """Devuelve los k chunks más parecidos a la consulta."""
        self._cargar_o_construir()
        _, ids = self._indice.search(_embeber([consulta]), min(k, len(self._chunks)))
        return [self._chunks[i] for i in ids[0] if i != -1]

    def buscar_varias(self, consultas: list[str], k: int = 1) -> list[str]:
        """Busca varias consultas y junta los chunks sin repetir, respetando el orden."""
        vistos, resultado = set(), []
        for consulta in consultas:
            for chunk in self.buscar(consulta, k):
                if chunk not in vistos:
                    vistos.add(chunk)
                    resultado.append(chunk)
        return resultado