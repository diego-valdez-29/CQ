import base64
import binascii
import os
import tempfile

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


def analizar_wav(ruta_temporal: str) -> dict:
    canal_caller, canal_callee, sr = cargar_canales(ruta_temporal)
    senales = [
        detector.analizar(canal_caller, canal_callee, sr)
        for detector in detectores
    ]
    resultado = fusion.combinar(senales)
    return {
        "is_synthetic": resultado.is_synthetic,
        "confidence": resultado.confidence,
    }


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
