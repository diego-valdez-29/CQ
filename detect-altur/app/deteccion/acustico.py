"""app/deteccion/acustico.py — Detector acústico de voz sintética."""
import numpy as np
import librosa
import librosa.feature

from app.dominio.modelos import SenalScore
from app.dominio.puertos import DetectorDeSenal

# INCIERTO: usando fallback MFCC porque no hay certeza del checkpoint exacto de AASIST.
# Detector heurístico basado en varianza/regularidad espectral de MFCCs con librosa.


class DetectorAcustico:
    """Detector de voz sintética basado en características acústicas.

    Analiza el CANAL_CALLER (canal 0): el caller entrante que hay que
    clasificar, no el canal 1 (agente de Altur, conocido y no evaluado).
    """

    SR_OBJETIVO = 16000
    DURACION_MINIMA_S = 0.3

    def __init__(self):
        self._umbral_varianza = 0.02
        self._umbral_regularidad = 0.75

    @classmethod
    def _preprocesar(cls, audio: np.ndarray, sr: int) -> tuple:
        """Paso 2: resamplea a 16000 Hz, normaliza a [-1, 1], mide duración."""
        if sr != cls.SR_OBJETIVO:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=cls.SR_OBJETIVO)

        audio = audio.astype(np.float32)
        pico = np.max(np.abs(audio))
        if pico > 0 and pico != 1.0:
            audio = audio / pico

        duracion_s = len(audio) / cls.SR_OBJETIVO
        return audio, duracion_s

    @classmethod
    def _verificar_condiciones_analisis(cls, audio: np.ndarray) -> bool:
        """Paso 5: descarta audio demasiado corto o en silencio total."""
        if len(audio) / cls.SR_OBJETIVO < cls.DURACION_MINIMA_S:
            return False
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > 1e-4

    @staticmethod
    def _extraer_caracteristicas(audio: np.ndarray) -> dict:
        """Extrae MFCCs y métricas crudas usadas en el score."""
        mfccs = librosa.feature.mfcc(y=audio, sr=DetectorAcustico.SR_OBJETIVO, n_mfcc=13)
        espectrograma = np.abs(librosa.stft(audio))
        return {
            "varianza_mfcc": float(np.var(mfccs)),
            "regularidad_espectral": float(np.mean(np.diff(espectrograma, axis=1) == 0)),
        }

    def _score_heuristico(self, features: dict) -> float:
        """Paso 4: mapea las métricas crudas a un score [0, 1] donde 1.0 = sintético."""
        v_mfcc = features["varianza_mfcc"]
        r_espec = features["regularidad_espectral"]

        exponente = -(
            (v_mfcc - self._umbral_varianza) / self._umbral_varianza
            + (r_espec - self._umbral_regularidad) / self._umbral_regularidad
        )
        score = 1.0 / (1.0 + np.exp(exponente))
        return float(np.clip(score, 0.0, 1.0))

    def analizar(self, canal_caller: np.ndarray,
                 canal_callee: np.ndarray, sr: int) -> SenalScore:
        """Paso 3 y 6: inferencia y ensamblado final del score.

        Usa canal_caller — el caller entrante a clasificar — no canal_callee
        (canal 1, agente de Altur).
        """
        audio, duracion_s = self._preprocesar(canal_caller, sr)

        if not self._verificar_condiciones_analisis(audio):
            return SenalScore(
                nombre="acustico",
                score=0.5,
                detalle={
                    "error": "audio_insuficiente",
                    "duracion_s": duracion_s,
                    "sr_objetivo": DetectorAcustico.SR_OBJETIVO,
                },
            )

        features = self._extraer_caracteristicas(audio)
        score = self._score_heuristico(features)

        return SenalScore(
            nombre="acustico",
            score=score,
            detalle={
                "metodo": "fallback_mfcc",
                "varianza_mfcc": features["varianza_mfcc"],
                "regularidad_espectral": features["regularidad_espectral"],
                "duracion_s": duracion_s,
                "sr_objetivo": DetectorAcustico.SR_OBJETIVO,
            },
        )