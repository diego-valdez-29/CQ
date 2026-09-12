"""app/deteccion/comportamiento.py — Detector de consistencia de recuperacion."""
import librosa
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from app.dominio.modelos import SenalScore


class DetectorComportamiento:
    """Detector basado en la consistencia con la que el caller se recupera
    de eventos de interrupcion provocados por el agente.

    El AGENTE (canal_callee, canal 1) provoca a proposito momentos de
    interrupcion, silencio o atropello. Un humano se recupera de esos
    eventos de forma instantanea pero errática (alta desviacion estandar
    entre tiempos de recuperacion); una maquina se recupera de forma
    consistente (baja desviacion), y esa consistencia es la señal de
    sospecha de sintetico.
    """

    SR_VAD = 16000
    UMBRAL_SILENCIO_S = 0.6  # duracion minima para considerar "silencio largo repentino"
    VENTANA_BUSQUEDA_RECUPERACION_S = 5.0  # tope de busqueda de la siguiente habla del caller
    ESCALA_NORMALIZACION_STD = 0.3  # INCIERTO: sin calibrar con dataset real de Altur

    def __init__(self):
        self._modelo = None

    def _obtener_modelo(self):
        if self._modelo is None:
            self._modelo = load_silero_vad()
        return self._modelo

    def _resamplear(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = audio.astype(np.float32)
        if sr == self.SR_VAD:
            return audio
        return librosa.resample(audio, orig_sr=sr, target_sr=self.SR_VAD)

    def _timestamps_habla(self, audio: np.ndarray) -> list[dict]:
        tensor = torch.from_numpy(audio)
        return get_speech_timestamps(
            tensor,
            self._obtener_modelo(),
            sampling_rate=self.SR_VAD,
            return_seconds=True,
        )

    def _detectar_eventos(
        self, habla_callee: list[dict], habla_caller: list[dict]
    ) -> list[float]:
        """Detecta instantes (en segundos) donde termina un evento de
        interrupcion/silencio/atropello provocado por el agente (canal_callee).

        # INCIERTO: silero-vad no distingue "silencio provocado a proposito
        # por el agente" de silencio normal entre turnos de conversacion.
        # Usamos como proxy: (a) cualquier gap de silencio >= UMBRAL_SILENCIO_S
        # entre segmentos de habla del callee, o (b) solapamiento entre habla
        # del callee y habla del caller (atropello), y medimos cuanto tarda el
        # caller en retomar el habla despues de cada uno.
        """
        eventos = []

        for anterior, siguiente in zip(habla_callee, habla_callee[1:]):
            gap = siguiente["start"] - anterior["end"]
            if gap >= self.UMBRAL_SILENCIO_S:
                eventos.append(anterior["end"])

        for seg_callee in habla_callee:
            for seg_caller in habla_caller:
                solapa = (
                    seg_callee["start"] < seg_caller["end"]
                    and seg_callee["end"] > seg_caller["start"]
                )
                if solapa:
                    eventos.append(seg_callee["end"])

        return sorted(set(eventos))

    def _tiempos_recuperacion(
        self, eventos: list[float], habla_caller: list[dict]
    ) -> list[float]:
        inicios_caller = [seg["start"] for seg in habla_caller]
        recuperaciones = []
        for t_evento in eventos:
            candidatos = [inicio for inicio in inicios_caller if inicio >= t_evento]
            if not candidatos:
                continue
            hueco = min(candidatos) - t_evento
            if hueco <= self.VENTANA_BUSQUEDA_RECUPERACION_S:
                recuperaciones.append(hueco)
        return recuperaciones

    def _score_heuristico(self, std_recuperacion: float) -> float:
        """Baja std (recuperacion uniforme) = mas sospechoso de maquina.

        Normalizacion lineal simple, acotada a [0, 1]. La escala aun no esta
        calibrada con el dataset real (ver ESCALA_NORMALIZACION_STD).
        """
        score = 1.0 - (std_recuperacion / self.ESCALA_NORMALIZACION_STD)
        return float(np.clip(score, 0.0, 1.0))

    def analizar(
        self,
        canal_caller: np.ndarray,
        canal_callee: np.ndarray,
        sr: int,
    ) -> SenalScore:
        audio_caller = self._resamplear(canal_caller, sr)
        audio_callee = self._resamplear(canal_callee, sr)

        habla_caller = self._timestamps_habla(audio_caller)
        habla_callee = self._timestamps_habla(audio_callee)

        eventos = self._detectar_eventos(habla_callee, habla_caller)
        recuperaciones = self._tiempos_recuperacion(eventos, habla_caller)

        if len(recuperaciones) < 2:
            return SenalScore(
                nombre="comportamiento",
                score=0.5,
                detalle={
                    "error": "eventos_insuficientes",
                    "numero_eventos_detectados": len(eventos),
                    "media_recuperacion": (
                        float(np.mean(recuperaciones)) if recuperaciones else None
                    ),
                    "std_recuperacion": None,
                },
            )

        media = float(np.mean(recuperaciones))
        std = float(np.std(recuperaciones))
        score = self._score_heuristico(std)

        return SenalScore(
            nombre="comportamiento",
            score=score,
            detalle={
                "metodo": "consistencia_recuperacion_silero_vad",
                "numero_eventos_detectados": len(eventos),
                "media_recuperacion": media,
                "std_recuperacion": std,
            },
        )
