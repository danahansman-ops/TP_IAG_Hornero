"""
evaluar.py: evaluación de la etapa 1 con varias corridas por consulta (Clase 4).

Un agente no se evalúa con una sola corrida: el mismo pedido puede salir bien una vez y mal
la siguiente. medir() repite cada consulta de prueba y cuenta en cuántas corridas funciona,
con las llamadas al LLM, los tokens y los segundos de cada una (ContadorUso de agentes.py).

Cómo correrlo (desde esta carpeta, con el venv activado):
    python evaluar.py                      # las 4 consultas de la consigna, 3 corridas cada una
    python evaluar.py --corridas 5 --consultas 3 4
    python evaluar.py --consultas fuera    # el caso límite fuera de alcance

Cada corrida parte del mismo historial: el CSV se restaura después de cada una, así una pieza
guardada en la corrida 1 no cambia lo que ve la corrida 2 (ni queda en el entregable).
Ojo con el límite diario de tokens de Groq: 4 consultas x 3 corridas consumen buena parte.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from agentes import ejecutar_marketing
from tools import RUTA_CONTENIDOS

# Funciona = la pieza se aprobó (veredicto APROBADA, checklist completo y sin fallas de los
# controles por código). Para el caso fuera de alcance, funciona si el sistema lo rechaza.
CONSULTAS = {
    "1": ("Armá un post de Instagram para clientes Premium anunciando la nueva colección de "
          "muebles de exterior.", "aprobada"),
    "2": ("Escribí un email para recuperar a clientes que tuvieron problemas con una entrega.",
          "aprobada"),
    "3": ("Proponé un mensaje de WhatsApp para clientes Nuevos con un 15% de descuento en Cocina "
          "y guardalo.", "aprobada"),
    "4": ("Armá un push para clientes Recurrentes sobre aires acondicionados que no repita lo que "
          "ya les mandamos.", "aprobada"),
    "fuera": ("¿Quién va a ganar el Mundial?", "fuera_de_alcance"),
}
ESPERA = 30   # segundos entre corridas, por el límite de tokens por minuto de Groq


def funciono(estado: dict, esperado: str) -> bool:
    if esperado == "fuera_de_alcance":
        return bool(estado.get("fuera_de_alcance"))
    return bool(estado.get("aprobada"))


def motivo(estado: dict) -> str:
    """Por qué no funcionó, en una línea (la primera observación del revisor)."""
    if estado.get("fuera_de_alcance"):
        return "fuera de alcance"
    if estado.get("sin_canal"):
        return "no se reconoció el canal"
    lineas = [l.strip(" -*") for l in str(estado.get("observaciones", "")).splitlines() if l.strip(" -*")]
    return lineas[0][:120] if lineas else ""


def medir(nombre: str, pedido: str, esperado: str, corridas: int = 3, espera: int = ESPERA) -> list[dict]:
    """Corre el mismo pedido varias veces y devuelve una fila por corrida."""
    print(f"\nConsulta {nombre}: {pedido}")
    csv_original = Path(RUTA_CONTENIDOS).read_bytes()
    filas = []
    for i in range(corridas):
        try:
            resultado = ejecutar_marketing(pedido)
            estado, uso = resultado["estado"], resultado["uso"]
            ok = funciono(estado, esperado)
            fila = {"Consulta": nombre, "Corrida": i + 1, "Funcionó": ok,
                    "Borradores": estado.get("borradores", 0),
                    "Llamadas al LLM": uso["llamadas"], "Tokens entrada": uso["tokens_entrada"],
                    "Tokens salida": uso["tokens_salida"], "Segundos": uso["segundos"],
                    "Motivo": "" if ok else motivo(estado)}
        except Exception as e:   # p. ej. el 429 de Groq: se registra y se sigue con la próxima
            fila = {"Consulta": nombre, "Corrida": i + 1, "Funcionó": False, "Borradores": 0,
                    "Llamadas al LLM": 0, "Tokens entrada": 0, "Tokens salida": 0, "Segundos": 0.0,
                    "Motivo": f"error: {str(e)[:100]}"}
        finally:
            Path(RUTA_CONTENIDOS).write_bytes(csv_original)
        filas.append(fila)
        estado_txt = "funcionó" if fila["Funcionó"] else f"NO funcionó ({fila['Motivo']})"
        print(f"  Corrida {i + 1}: {estado_txt} · {fila['Borradores']} borrador/es · "
              f"{fila['Llamadas al LLM']} llamadas · "
              f"{fila['Tokens entrada'] + fila['Tokens salida']} tokens · {fila['Segundos']} s")
        if i < corridas - 1:
            time.sleep(espera)
    aciertos = sum(f["Funcionó"] for f in filas)
    print(f"→ {aciertos}/{corridas} corridas funcionaron")
    return filas


def resumir(filas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(filas)
    df["Tokens totales"] = df["Tokens entrada"] + df["Tokens salida"]
    resumen = df.groupby("Consulta", sort=False).agg(
        Corridas=("Corrida", "count"), Funcionaron=("Funcionó", "sum"),
        **{"Borradores (prom.)": ("Borradores", "mean"),
           "Llamadas (prom.)": ("Llamadas al LLM", "mean"),
           "Tokens (prom.)": ("Tokens totales", "mean"),
           "Segundos (prom.)": ("Segundos", "mean"),
           "Segundos (máx.)": ("Segundos", "max")})
    resumen["Tasa de éxito"] = (resumen["Funcionaron"] / resumen["Corridas"]).map("{:.0%}".format)
    return resumen.round(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa la etapa 1 con varias corridas por consulta.")
    parser.add_argument("--corridas", type=int, default=3)
    parser.add_argument("--consultas", nargs="+", default=["1", "2", "3", "4"], choices=list(CONSULTAS))
    parser.add_argument("--espera", type=int, default=ESPERA)
    parser.add_argument("--salida", default="evaluacion_etapa1.csv",
                        help="CSV con una fila por corrida (para el informe)")
    args = parser.parse_args()

    filas = []
    for n, nombre in enumerate(args.consultas):
        pedido, esperado = CONSULTAS[nombre]
        filas += medir(nombre, pedido, esperado, args.corridas, args.espera)
        if n < len(args.consultas) - 1:
            time.sleep(args.espera)

    pd.DataFrame(filas).to_csv(args.salida, index=False)
    print("\nResumen:\n")
    print(resumir(filas).to_string())
    print(f"\nDetalle por corrida guardado en {args.salida}")
