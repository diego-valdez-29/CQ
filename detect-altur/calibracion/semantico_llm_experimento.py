"""calibracion/semantico_llm_experimento.py — cliente LLM (paso 2, la Spark)
que antes vivia en app/deteccion/semantico.py.

Movido aqui, sin usar en el pipeline de main.py, tras la calibracion final
(n=60): la señal semantica salio con peso 0 en Fusion en 3 calibraciones
independientes (ver README.md). Se probo tanto con este cliente (la Spark)
como con OpenAI gpt-4o-mini (calibracion/probar_semantico_openai.py) y ningun
LLM aporto separacion human/sintetico suficiente para justificar la latencia
y el riesgo de una llamada de red sincrona dentro de /detect. Queda como
referencia documentada de una señal explorada y descartada con evidencia
real, no un recorte por falta de tiempo.

DetectorSemanticoConLLM reconstruye el analizar() completo de antes del
recorte (paso 1 regex + paso 2 LLM con timeout de pared), reusando
_transcribir/_filtro_regex de la clase de produccion via herencia, para poder
reproducir o volver a probar la señal completa fuera de produccion.
"""
import concurrent.futures
import json
import os
import re

import numpy as np
import requests

from app.deteccion.semantico import DetectorSemantico
from app.dominio.modelos import SenalScore

SPARK_BASE_URL = "http://100.122.49.1:8080/api"
SPARK_MODEL = "uncensored"
SPARK_TIMEOUT_S = 2.0
TIMEOUT_PARED_S = 3

PROMPT_TEMPLATE = """
Eres un clasificador. Tu única salida es un JSON, nada más.

El siguiente es el transcript de un caller respondiendo a un agente de un banco.
El agente, a veces, pregunta deliberadamente sobre información que NO EXISTE
para el caller (una prueba). Un humano real responde algo como "no tengo esa
información" o "eso no lo sé". Un modelo de lenguaje sintético tiende a
INVENTAR una respuesta plausible en vez de admitir que no la tiene.

Ejemplos:
- Agente pregunta por un dato inventado, caller responde "no cuento con eso" -> humano
- Agente pregunta por un dato inventado, caller responde con un dato concreto
  (una fecha, un número, un nombre) que no fue mencionado antes -> sintético
- Caller pide que le repitan la pregunta o admite duda -> humano

Transcript a evaluar:
\"\"\"
{transcript}
\"\"\"

Responde SOLO este JSON, sin texto antes o después:
{{"score": <float 0.0-1.0, 1.0=muy probablemente sintetico>, "razon": "<una frase>"}}
"""


class DetectorSemanticoConLLM(DetectorSemantico):
    """Igual que DetectorSemantico, pero con el paso 2 (LLM de la Spark)
    restaurado. No se usa en main.py — solo para reproducir la señal
    completa en un script de calibracion/experimento.
    """

    @staticmethod
    def _extraer_json(texto: str) -> dict:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if not match:
            raise ValueError("no se encontro un JSON en la respuesta del LLM")
        return json.loads(match.group(0))

    def _clasificar_con_llm(self, transcript: str) -> SenalScore:
        prompt = PROMPT_TEMPLATE.format(transcript=transcript)
        try:
            headers = {"Authorization": f"Bearer {os.environ['SPARK_API_KEY']}"}
            respuesta = requests.post(
                f"{SPARK_BASE_URL}/chat/completions",
                json={
                    "model": SPARK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers=headers,
                timeout=SPARK_TIMEOUT_S,
            )
            respuesta.raise_for_status()
            # El campo "reasoning" es el razonamiento interno del modelo y NUNCA
            # contiene el JSON pedido: el JSON siempre esta en "content".
            contenido = respuesta.json()["choices"][0]["message"]["content"]
            parsed = self._extraer_json(contenido)
            score = float(np.clip(float(parsed["score"]), 0.0, 1.0))
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            ValueError,
            TypeError,
        ):
            return SenalScore(
                nombre="semantico",
                score=0.5,
                detalle={"error": "llm_no_disponible"},
            )

        return SenalScore(
            nombre="semantico",
            score=score,
            detalle={
                "metodo": "llm_spark",
                "razon": parsed.get("razon"),
            },
        )

    def analizar(
        self,
        canal_caller: np.ndarray,
        canal_callee: np.ndarray,
        sr: int,
    ) -> SenalScore:
        transcript = self._transcribir(canal_caller, sr)

        if not transcript:
            return SenalScore(
                nombre="semantico",
                score=0.5,
                detalle={"error": "transcript_vacio"},
            )

        match = self._filtro_regex(transcript)
        if match is not None:
            return SenalScore(
                nombre="semantico",
                score=0.15,
                detalle={"metodo": "regex_honesto", "match": match},
            )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._clasificar_con_llm, transcript)
        try:
            return future.result(timeout=TIMEOUT_PARED_S)
        except concurrent.futures.TimeoutError:
            return SenalScore(
                nombre="semantico",
                score=0.5,
                detalle={"fallback": "timeout_pared"},
            )
        finally:
            executor.shutdown(wait=False)
