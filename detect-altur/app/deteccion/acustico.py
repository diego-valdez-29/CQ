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

    # Media/desviación estándar reales de varianza_mfcc y regularidad_espectral,
    # calculadas sobre n=30 llamadas reales del manifest (train, semilla=0,
    # 15 human + 15 synthetic, 8kHz telefonia) con calibrar_acustico_crudo.py.
    # Población completa (no separada por clase): en producción no se conoce
    # la clase de antemano, así que el z-score usa estos estadísticos globales.
    MEDIA_VARIANZA = 16099.42
    STD_VARIANZA = 2670.55
    MEDIA_REGULARIDAD = 0.03622
    STD_REGULARIDAD = 0.10415

    def __init__(self):
        self._media_varianza = self.MEDIA_VARIANZA
        self._std_varianza = self.STD_VARIANZA
        self._media_regularidad = self.MEDIA_REGULARIDAD
        self._std_regularidad = self.STD_REGULARIDAD

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

    def _calcular_z_scores(self, features: dict) -> tuple[float, float]:
        z_varianza = (features["varianza_mfcc"] - self._media_varianza) / self._std_varianza
        z_regularidad = (
            (features["regularidad_espectral"] - self._media_regularidad)
            / self._std_regularidad
        )
        return z_varianza, z_regularidad

    def _calcular_exponente(self, features: dict) -> float:
        z_varianza, z_regularidad = self._calcular_z_scores(features)
        return -(z_varianza + z_regularidad)

    def _score_heuristico(self, features: dict) -> float:
        """Paso 4: mapea las métricas crudas a un score [0, 1] donde 1.0 = sintético.

        INCIERTO: en n=30 llamadas reales (ver calibrar_acustico_crudo.py),
        varianza_mfcc no separa human/synthetic (medias a 0.4% de distancia,
        muy por debajo del ruido intra-clase) y regularidad_espectral separa
        débilmente y de forma inestable (dominado por dos outliers humanos).
        Esta normalización por z-score corrige la saturación en 1.0 que
        causaba el umbral fijo anterior, pero la señal resultante sigue
        siendo débil — no tratar "acustico" como decisiva por sí sola.
        """
        exponente = self._calcular_exponente(features)
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
                # INCIERTO: varianza_mfcc y regularidad_espectral muestran
                # separacion nula/debil entre human y synthetic en n=30
                # llamadas reales de 8kHz - señal secundaria, no confiar en
                # ella como decisiva.
                "separacion_debil": True,
            },
        )

    def analizar_debug(self, canal_caller: np.ndarray,
                        canal_callee: np.ndarray, sr: int) -> tuple[SenalScore, dict]:
        """Igual que analizar(), pero ademas regresa un dict de diagnostico con
        los valores crudos que entran al sigmoid de _score_heuristico. Opt-in,
        no lo llama el pipeline normal (main.py) — solo para depurar por que
        el score satura.
        """
        senal = self.analizar(canal_caller, canal_callee, sr)

        audio, _ = self._preprocesar(canal_caller, sr)
        if not self._verificar_condiciones_analisis(audio):
            return senal, {"error": "audio_insuficiente"}

        features = self._extraer_caracteristicas(audio)
        z_varianza, z_regularidad = self._calcular_z_scores(features)
        exponente = self._calcular_exponente(features)

        debug = {
            "varianza_mfcc": features["varianza_mfcc"],
            "media_varianza": self._media_varianza,
            "std_varianza": self._std_varianza,
            "z_varianza": z_varianza,
            "regularidad_espectral": features["regularidad_espectral"],
            "media_regularidad": self._media_regularidad,
            "std_regularidad": self._std_regularidad,
            "z_regularidad": z_regularidad,
            "exponente": exponente,
            "score_final": senal.score,
        }
        return senal, debug