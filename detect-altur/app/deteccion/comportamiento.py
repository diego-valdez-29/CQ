"""app/deteccion/comportamiento.py — Detector de consistencia de recuperacion."""
import librosa
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from app.dominio.modelos import SenalScore


class DetectorComportamiento:
    """Detector basado en tiempo_primera_habla: el timestamp (segundos) en
    el que canal_caller habla por primera vez en toda la llamada.

    En n=30 llamadas reales (ver calibrar_comportamiento_crudo.py) esta
    señal sola separa human/synthetic con |diff medias|=0.196 y
    accuracy=76.7%: humanos tardan en promedio ~7.5s (bimodal: unos
    responden rapido en 1-4s, otros lento en 8-10s), sinteticos tardan
    ~10.2s y estan mucho mas agrupados (9.2-12.0s). Un tiempo alto es
    señal de sospecha de sintetico.

    std_recuperacion (consistencia con la que el caller se recupera de
    eventos de interrupcion/silencio/atropello provocados por el AGENTE,
    canal_callee) tambien se calcula y se reporta en detalle como
    evidencia adicional, pero NO participa en el score numerico: en las
    mismas 30 llamadas muestra separacion nula o invertida por si sola, y
    combinarla con tiempo_primera_habla por promedio de z-scores diluyo la
    señal fuerte (|diff medias| bajo de 0.196 a 0.080, accuracy de 76.7% a
    60.0%) en vez de reforzarla. Ver _score_heuristico.
    """

    SR_VAD = 16000
    UMBRAL_SILENCIO_S = 0.6  # duracion minima para considerar "silencio largo repentino"
    VENTANA_BUSQUEDA_RECUPERACION_S = 5.0  # tope de busqueda de la siguiente habla del caller

    # Media/desviacion estandar reales de tiempo_primera_habla, calculadas
    # sobre n=30 llamadas reales del manifest (train, semilla=0, 15 human +
    # 15 synthetic) con calibrar_comportamiento_crudo.py. Poblacion completa
    # (no separada por clase): en produccion no se conoce la clase de
    # antemano, asi que el z-score usa estos estadisticos globales, mismo
    # patron que DetectorAcustico._calcular_z_scores.
    MEDIA_TIEMPO_PRIMERA_HABLA = 8.82000
    STD_TIEMPO_PRIMERA_HABLA = 2.69003

    def __init__(self):
        self._modelo = None
        self._media_tiempo_primera_habla = self.MEDIA_TIEMPO_PRIMERA_HABLA
        self._std_tiempo_primera_habla = self.STD_TIEMPO_PRIMERA_HABLA
        # Pre-cargar el modelo en __init__ para evitar cold-start en la primera peticion HTTP
        self._obtener_modelo()

    def _obtener_modelo(self):
        if self._modelo is None:
            self._modelo = load_silero_vad()
            try:
                dummy_audio = torch.zeros(16000, dtype=torch.float32)
                get_speech_timestamps(dummy_audio, self._modelo, sampling_rate=16000)
            except Exception:
                pass
        return self._modelo

    def _resamplear(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = audio.astype(np.float32)
        if sr == self.SR_VAD:
            return audio
        if sr == 8000:
            return np.repeat(audio, 2)
        return librosa.resample(audio, orig_sr=sr, target_sr=self.SR_VAD)

    def _timestamps_habla(self, audio: np.ndarray) -> list[dict]:
        max_muestras = int(35.0 * self.SR_VAD)
        tensor = torch.from_numpy(audio[:max_muestras])
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

    def _tiempo_primera_habla(self, habla_caller: list[dict]) -> float | None:
        """Timestamp (segundos) del inicio del primer segmento de habla de
        canal_caller en toda la llamada. No requiere VAD adicional: ya sale
        de _timestamps_habla.
        """
        if not habla_caller:
            return None
        return float(habla_caller[0]["start"])

    def _calcular_z_score(self, tiempo_primera_habla: float) -> float:
        return (
            (tiempo_primera_habla - self._media_tiempo_primera_habla)
            / self._std_tiempo_primera_habla
        )

    def _calcular_exponente(self, tiempo_primera_habla: float) -> float:
        """z negado: tiempo_primera_habla por ENCIMA de la media (z
        positivo) es sospechoso de maquina (sinteticos tardan mas en hablar
        por primera vez), asi que se niega para que el exponente sea
        negativo en ese caso y el sigmoid suba el score hacia 1.0.
        """
        return -self._calcular_z_score(tiempo_primera_habla)

    def _score_heuristico(self, tiempo_primera_habla: float) -> float:
        """tiempo_primera_habla alto = mas sospechoso de maquina.

        Score basado UNICAMENTE en tiempo_primera_habla, normalizado por
        z-score poblacional (mismo patron que
        DetectorAcustico._score_heuristico).

        # DECISION: std_recuperacion se probo combinada por promedio de
        # z-scores ((z_std - z_tiempo) / 2) y diluyo la señal fuerte de
        # tiempo_primera_habla: sola, tiempo_primera_habla da
        # |diff medias|=0.196 y accuracy=76.7% (n=30, ver
        # calibrar_comportamiento_crudo.py); combinada con std_recuperacion
        # cae a |diff medias|=0.080 y accuracy=60.0%, porque std_recuperacion
        # muestra separacion nula o invertida por si sola. Por eso el score
        # final usa solo tiempo_primera_habla; std_recuperacion se sigue
        # calculando y reportando en detalle como evidencia adicional, pero
        # no entra al calculo numerico.
        """
        exponente = self._calcular_exponente(tiempo_primera_habla)
        score = 1.0 / (1.0 + np.exp(exponente))
        return float(np.clip(score, 0.0, 1.0))

    def analizar(
        self,
        canal_caller: np.ndarray,
        canal_callee: np.ndarray,
        sr: int,
    ) -> SenalScore:
        audio_caller = self._resamplear(canal_caller, sr)
        habla_caller = self._timestamps_habla(audio_caller)
        tiempo_primera_habla = self._tiempo_primera_habla(habla_caller)

        if tiempo_primera_habla is None:
            return SenalScore(
                nombre="comportamiento",
                score=0.5,
                detalle={
                    "error": "habla_caller_no_detectada",
                    "tiempo_primera_habla": None,
                },
            )

        score = self._score_heuristico(tiempo_primera_habla)

        return SenalScore(
            nombre="comportamiento",
            score=score,
            detalle={
                "metodo": "tiempo_primera_habla_silero_vad",
                "tiempo_primera_habla": tiempo_primera_habla,
            },
        )
