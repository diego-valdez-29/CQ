"""app/deteccion/acustico_wav2vec.py — Detector acustico de voz sintetica
basado en un modelo wav2vec2 preentrenado de deteccion de deepfakes.

Alternativa experimental a DetectorAcustico (MFCC + z-score), para comparar
con datos reales antes de decidir si reemplaza al detector actual. No se usa
todavia en el pipeline de main.py.
"""
import numpy as np
import librosa
from transformers import pipeline

from app.dominio.modelos import SenalScore


class DetectorAcusticoWav2Vec:
    """Detector de voz sintetica basado en el modelo
    MelodyMachine/Deepfake-audio-detection-V2 (wav2vec2 fine-tuneado para
    clasificacion binaria fake/real).

    Analiza el CANAL_CALLER (canal 0): el caller entrante que hay que
    clasificar, no el canal 1 (agente de Altur, conocido y no evaluado).
    """

    MODELO = "MelodyMachine/Deepfake-audio-detection-V2"
    DURACION_MINIMA_S = 0.3

    # Etiquetas exactas que devuelve el modelo (confirmado via
    # AutoConfig.from_pretrained(MODELO).id2label): {0: "fake", 1: "real"}.
    # No asumir human/synthetic ni el orden — el pipeline regresa una lista
    # de dicts {"label", "score"} para las dos clases, en orden de score
    # descendente, no en orden fijo de id.
    ETIQUETA_SINTETICO = "fake"

    def __init__(self):
        self._clasificador = pipeline("audio-classification", model=self.MODELO)
        # sampling_rate real que espera el feature extractor del modelo,
        # no asumido — confirmado en 16000 via
        # AutoFeatureExtractor.from_pretrained(MODELO).sampling_rate, pero
        # se lee del propio pipeline por si el modelo cambia.
        self._sr_modelo = self._clasificador.feature_extractor.sampling_rate

    def _preprocesar(self, audio: np.ndarray, sr: int) -> tuple:
        if sr != self._sr_modelo:
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=sr, target_sr=self._sr_modelo
            )
        audio = audio.astype(np.float32)
        pico = np.max(np.abs(audio))
        if pico > 0 and pico != 1.0:
            audio = audio / pico
        duracion_s = len(audio) / self._sr_modelo
        return audio, duracion_s

    def _verificar_condiciones_analisis(self, audio: np.ndarray) -> bool:
        if len(audio) / self._sr_modelo < self.DURACION_MINIMA_S:
            return False
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > 1e-4

    def _score_desde_salida(self, salida: list[dict]) -> float:
        """Convierte la lista de {"label", "score"} del pipeline a un score
        0..1 donde 1.0 = sintetico, tomando la probabilidad de la etiqueta
        "fake" (no asumir orden ni cantidad de etiquetas devueltas).
        """
        for resultado in salida:
            if resultado["label"] == self.ETIQUETA_SINTETICO:
                return float(np.clip(resultado["score"], 0.0, 1.0))
        raise ValueError(f"etiqueta '{self.ETIQUETA_SINTETICO}' no encontrada en salida: {salida}")

    def analizar(self, canal_caller: np.ndarray,
                 canal_callee: np.ndarray, sr: int) -> SenalScore:
        try:
            audio, duracion_s = self._preprocesar(canal_caller, sr)

            if not self._verificar_condiciones_analisis(audio):
                return SenalScore(
                    nombre="acustico",
                    score=0.5,
                    detalle={
                        "error": "audio_insuficiente",
                        "duracion_s": duracion_s,
                        "sr_modelo": self._sr_modelo,
                    },
                )

            salida = self._clasificador({"array": audio, "sampling_rate": self._sr_modelo})
            score = self._score_desde_salida(salida)

            return SenalScore(
                nombre="acustico",
                score=score,
                detalle={
                    "metodo": "wav2vec2_deepfake_audio_detection_v2",
                    "modelo": self.MODELO,
                    "salida_cruda": salida,
                    "duracion_s": duracion_s,
                    "sr_modelo": self._sr_modelo,
                },
            )
        except Exception as exc:
            return SenalScore(
                nombre="acustico",
                score=0.5,
                detalle={
                    "error": f"fallo_inferencia: {exc!r}",
                    "modelo": self.MODELO,
                },
            )
