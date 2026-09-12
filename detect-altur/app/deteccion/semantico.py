"""app/deteccion/semantico.py — Detector semantico: honestidad vs confabulacion."""
import concurrent.futures
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
LLM_TIMEOUT_PARED_S = 3.0

WHISPER_TIMEOUT_MIN_S = 25.0
WHISPER_TIMEOUT_MAX_S = 90.0
WHISPER_TIMEOUT_FACTOR = 0.6

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

    PASO 3: si Whisper/LLM falla, da timeout o no se puede parsear, no truena:
    score=0.5 con detalle de fallback.
    """

    SR_WHISPER = 16000
    WHISPER_MODEL_SIZE = "small"
    WHISPER_IDIOMA = "es"

    def __init__(self):
        self._modelo_whisper = None
        self._patrones = [re.compile(p, re.IGNORECASE) for p in PATRONES_HONESTOS]
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _obtener_whisper(self) -> WhisperModel:
        if self._modelo_whisper is None:
            self._modelo_whisper = WhisperModel(
                self.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
            )
        return self._modelo_whisper

    @staticmethod
    def _timeout_whisper_s(duracion_s: float) -> float:
        return min(WHISPER_TIMEOUT_MAX_S, max(WHISPER_TIMEOUT_MIN_S, duracion_s * WHISPER_TIMEOUT_FACTOR))

    def _recrear_executor(self) -> None:
        self._executor.shutdown(wait=False)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _ejecutar_con_timeout(self, fn, timeout_s: float, *args):
        future = self._executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            self._recrear_executor()
            raise

    def _preparar_audio(self, canal_caller: np.ndarray, sr: int) -> np.ndarray:
        audio = canal_caller.astype(np.float32)
        if sr != self.SR_WHISPER:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SR_WHISPER)
        return audio

    def _transcribir_sin_timeout(self, audio: np.ndarray) -> list[dict]:
        modelo = self._obtener_whisper()
        segmentos, _ = modelo.transcribe(
            audio,
            language=self.WHISPER_IDIOMA,
            beam_size=1,
            vad_filter=True,
        )
        return [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
            for seg in segmentos
        ]

    def _transcribir(self, canal_caller: np.ndarray, sr: int) -> list[dict] | None:
        """Transcribe el caller. Devuelve segmentos {start, end, text} o None si timeout."""
        audio = self._preparar_audio(canal_caller, sr)
        duracion_s = len(audio) / self.SR_WHISPER
        try:
            return self._ejecutar_con_timeout(
                self._transcribir_sin_timeout,
                self._timeout_whisper_s(duracion_s),
                audio,
            )
        except concurrent.futures.TimeoutError:
            return None

    @staticmethod
    def _texto_hasta(segmentos: list[dict], hasta_s: float | None = None) -> str:
        partes = [
            seg["text"]
            for seg in segmentos
            if seg["text"] and (hasta_s is None or seg["start"] < hasta_s)
        ]
        return " ".join(partes).strip()

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

    def _clasificar_transcript(self, transcript: str) -> SenalScore:
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

        try:
            return self._ejecutar_con_timeout(
                self._clasificar_con_llm,
                LLM_TIMEOUT_PARED_S,
                transcript,
            )
        except concurrent.futures.TimeoutError:
            return SenalScore(
                nombre="semantico",
                score=0.5,
                detalle={"fallback": "timeout_pared"},
            )

    def analizar(
        self,
        canal_caller: np.ndarray,
        canal_callee: np.ndarray,
        sr: int,
    ) -> SenalScore:
        segmentos = self._transcribir(canal_caller, sr)
        if segmentos is None:
            return SenalScore(
                nombre="semantico",
                score=0.5,
                detalle={"fallback": "timeout_whisper"},
            )
        return self._clasificar_transcript(self._texto_hasta(segmentos))
