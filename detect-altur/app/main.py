import base64
import binascii
import os
import tempfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

from app.deteccion.acustico import DetectorAcustico
from app.deteccion.comportamiento import DetectorComportamiento
from app.deteccion.fusion import Fusion
from app.deteccion.semantico import DetectorSemantico

app = FastAPI(title="detect-altur")

detectores = [
    DetectorAcustico(),
    DetectorComportamiento(),
    DetectorSemantico(),
]

fusion = Fusion(
    pesos={"acustico": 1.0, "comportamiento": 1.0, "semantico": 1.0},
    umbral=0.5,
)

# Checkpoints de decision secuencial, calibrados contra llamadas reales del
# dataset (manifest.csv: duracion minima real = 61s). Si la confidence en un
# checkpoint intermedio supera el umbral, se responde sin esperar mas audio;
# el checkpoint final (fin de llamada) siempre decide, sin importar la confidence.
CHECKPOINTS_S = (25.0, 40.0)
UMBRAL_DECISION_TEMPRANA = 0.75  # punto de partida, ajustable

# Nombre de campo del WAV en base64 aun no confirmado con los organizadores:
# aceptamos varios candidatos hasta que el juez real nos diga cual usa.
CAMPOS_BASE64_POSIBLES = ("audio_base64", "audio", "wav_base64", "data")


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    audio_base64: str | None = None
    audio: str | None = None
    wav_base64: str | None = None
    data: str | None = None


def cargar_canales(ruta_audio: str):
    data, sr = sf.read(ruta_audio, always_2d=True)
    if data.shape[1] == 1:
        canal_caller = data[:, 0]
        canal_callee = data[:, 0]
    else:
        canal_caller = data[:, 0]
        canal_callee = data[:, 1]
    return canal_caller, canal_callee, sr


def _evaluar(canal_caller: np.ndarray, canal_callee: np.ndarray, sr: int):
    senales = [
        detector.analizar(canal_caller, canal_callee, sr)
        for detector in detectores
    ]
    return fusion.combinar(senales)


def _respuesta(resultado, checkpoint: str, duracion_audio_usada_s: float, duracion_total_s: float) -> dict:
    return {
        "is_synthetic": resultado.is_synthetic,
        "confidence": resultado.confidence,
        "detalle": {
            "checkpoint": checkpoint,
            "duracion_audio_usada_s": duracion_audio_usada_s,
            "duracion_total_llamada_s": duracion_total_s,
            "umbral_decision_temprana": UMBRAL_DECISION_TEMPRANA,
        },
    }


def decidir_secuencial(canal_caller: np.ndarray, canal_callee: np.ndarray, sr: int) -> dict:
    duracion_total_s = len(canal_caller) / sr

    for checkpoint_s in CHECKPOINTS_S:
        if duracion_total_s <= checkpoint_s:
            continue  # no hay suficiente audio todavia para este checkpoint

        n_muestras = int(checkpoint_s * sr)
        resultado = _evaluar(canal_caller[:n_muestras], canal_callee[:n_muestras], sr)
        if resultado.confidence >= UMBRAL_DECISION_TEMPRANA:
            return _respuesta(resultado, f"{checkpoint_s:.0f}s", checkpoint_s, duracion_total_s)

    resultado = _evaluar(canal_caller, canal_callee, sr)
    return _respuesta(resultado, "fin_de_llamada", duracion_total_s, duracion_total_s)


def analizar_wav(ruta_temporal: str) -> dict:
    canal_caller, canal_callee, sr = cargar_canales(ruta_temporal)
    return decidir_secuencial(canal_caller, canal_callee, sr)


@app.post("/detect")
async def detect(body: DetectRequest):
    audio_base64 = None
    for campo in CAMPOS_BASE64_POSIBLES:
        valor = getattr(body, campo, None)
        if valor:
            audio_base64 = valor
            break

    if audio_base64 is None:
        keys_recibidas = sorted(
            set(body.model_fields_set) | set((body.model_extra or {}).keys())
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_se_encontro_campo_base64_valido",
                "campos_aceptados": list(CAMPOS_BASE64_POSIBLES),
                "keys_recibidas": keys_recibidas,
            },
        )

    try:
        audio_bytes = base64.b64decode(audio_base64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"base64 invalido: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        ruta_temporal = tmp.name

    try:
        return analizar_wav(ruta_temporal)
    finally:
        os.remove(ruta_temporal)


@app.post("/detect-dev")
async def detect_dev(file: UploadFile):
    """Solo para desarrollo local con curl/multipart. No es la ruta que califican."""
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        ruta_temporal = tmp.name

    try:
        return analizar_wav(ruta_temporal)
    finally:
        os.remove(ruta_temporal)
