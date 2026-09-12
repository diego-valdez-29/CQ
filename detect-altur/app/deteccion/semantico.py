"""app/deteccion/semantico.py — Detector semantico: honestidad vs confabulacion."""
import json
import os
import re

import librosa
import numpy as np
import requests
from faster_whisper import WhisperModel

from app.dominio.modelos import SenalScore

SPARK_BASE_URL = "http://100.122.49.1:8080/api"
SPARK_MODEL = "uncensored"
SPARK_TIMEOUT_S = 2.0

PATRONES_HONESTOS = [
    r"no\s+s[ée]\b",
    r"no\s+tengo\s+(esa\s+)?informaci[oó]n",
    r"no\s+cuento\s+con\s+(eso|esa\s+informaci[oó]n)",
    r"no\s+recuerdo",
    r"no\s+me\s+suena",
    r"no\s+tengo\s+idea",
]

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


class DetectorSemantico:
    """Detector de confabulacion semantica ante preguntas sin informacion real.

    PASO 1: filtro regex rapido sobre el transcript de canal_caller buscando
    frases de honestidad ("no se", "no tengo esa informacion", etc.) - si hay
    match claro, se resuelve sin llamar al LLM (score=0.15, humano).

    PASO 2: si el regex no encuentra nada (indeterminado/posible confabulacion),
    se clasifica con el LLM de la Spark.

    PASO 3: si el LLM falla, da timeout o no se puede parsear, no truena:
    score=0.5 con detalle={"error": "llm_no_disponible"}.
    """

    SR_WHISPER = 16000
    WHISPER_MODEL_SIZE = "small"
    WHISPER_IDIOMA = "es"

    def __init__(self):
        self._modelo_whisper = None
        self._patrones = [re.compile(p, re.IGNORECASE) for p in PATRONES_HONESTOS]

    def _obtener_whisper(self) -> WhisperModel:
        if self._modelo_whisper is None:
            self._modelo_whisper = WhisperModel(
                self.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
            )
        return self._modelo_whisper

    def _transcribir(self, canal_caller: np.ndarray, sr: int) -> str:
        audio = canal_caller.astype(np.float32)
        if sr != self.SR_WHISPER:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SR_WHISPER)

        modelo = self._obtener_whisper()
        segmentos, _ = modelo.transcribe(audio, language=self.WHISPER_IDIOMA)
        return " ".join(seg.text.strip() for seg in segmentos).strip()

    def _filtro_regex(self, transcript: str) -> str | None:
        for patron in self._patrones:
            match = patron.search(transcript)
            if match:
                return match.group(0)
        return None

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

        return self._clasificar_con_llm(transcript)
