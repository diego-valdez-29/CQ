"""app/deteccion/semantico.py — Detector semantico: honestidad vs confabulacion."""
import re

import librosa
import numpy as np
from faster_whisper import WhisperModel

from app.dominio.modelos import SenalScore

PATRONES_HONESTOS = [
    r"no\s+s[ée]\b",
    r"no\s+tengo\s+(esa\s+)?informaci[oó]n",
    r"no\s+cuento\s+con\s+(eso|esa\s+informaci[oó]n)",
    r"no\s+recuerdo",
    r"no\s+me\s+suena",
    r"no\s+tengo\s+idea",
]


class DetectorSemantico:
    """Detector de confabulacion semantica ante preguntas sin informacion real.

    PASO 1: filtro regex rapido sobre el transcript de canal_caller buscando
    frases de honestidad ("no se", "no tengo esa informacion", etc.) - si hay
    match claro, se resuelve sin llamar a ningun LLM (score=0.15, humano).

    PASO 2 (ELIMINADO tras la calibracion final, n=60): antes, si el regex no
    resolvia, se clasificaba con un LLM (la Spark; tambien se probo OpenAI
    gpt-4o-mini en calibracion/probar_semantico_openai.py). En 3 calibraciones
    independientes con datos reales, la señal semantica salio con **peso 0**
    en Fusion — no aporto separacion humano/sintetico suficiente para
    justificar la latencia y el riesgo de una llamada de red sincrona dentro
    de /detect. Es una señal explorada y descartada con evidencia (ver
    README.md, seccion de calibracion), no un recorte por falta de tiempo.
    El cliente LLM completo (prompt, parseo, timeout de pared) se conserva sin
    usar como referencia en calibracion/semantico_llm_experimento.py.

    Cuando el regex no resuelve, se retorna directamente el score neutro
    (score=0.5, detalle {"metodo": "sin_llm_peso_cero"}) sin tocar la red.
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

        return SenalScore(
            nombre="semantico",
            score=0.5,
            detalle={"metodo": "sin_llm_peso_cero"},
        )
