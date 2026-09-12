import base64
import binascii
import json
import logging
import os
import tempfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.deteccion.comportamiento import DetectorComportamiento
from app.deteccion.fusion import Fusion

app = FastAPI(title="detect-altur")

# Habilitar CORS para permitir peticiones desde el frontend web y dominios externos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# DetectorAcustico y DetectorSemantico tienen peso 0.0 en Fusion, confirmado
# en tres calibraciones independientes con datos reales (ver README.md) -- no
# se llaman aqui. Sus clases se conservan intactas en sus archivos
# (app/deteccion/acustico.py, app/deteccion/semantico.py) como señales
# exploradas y descartadas con evidencia, no borradas.
detectores = [
    DetectorComportamiento(),
]

fusion = Fusion(
    # umbral recalibrado de 0.56 a 0.58: un barrido sobre split val (71
    # llamadas que nunca participaron en la calibracion original de pesos ni
    # umbral) encontro que 0.58 maximiza accuracy sobre val (0.789 vs 0.732
    # con 0.56) -- ver calibracion/validar_val.py y README.md ("Estado
    # honesto de calibracion").
    pesos={"acustico": 0.0, "comportamiento": 1.0, "semantico": 0.0},
    umbral=0.58,
)

# Checkpoints de decision secuencial, calibrados contra llamadas reales del
# dataset (manifest.csv: duracion minima real = 61s). Si la confidence en un
# checkpoint intermedio supera el umbral, se responde sin esperar mas audio;
# el checkpoint final (fin de llamada) siempre decide, sin importar la confidence.
CHECKPOINTS_S = (25.0, 40.0)
UMBRAL_DECISION_TEMPRANA = 0.75  # punto de partida, ajustable

# Nombre de campo del WAV en base64 aun no confirmado con los organizadores:
# aceptamos varios candidatos hasta que el juez real nos diga cual usa,
# buscando case-insensitive en las keys del body (no solo el nombre exacto).
CAMPOS_BASE64_POSIBLES = (
    "audio_base64",
    "audio",
    "wav_base64",
    "wav",
    "data",
    "file",
    "audioBase64",
    "audio_data",
    "base64_audio",
    "file_base64",
)


def _buscar_campo_base64(payload: dict) -> str | None:
    """Busca en payload, case-insensitive, la primera key que coincida con
    algun nombre de CAMPOS_BASE64_POSIBLES (en ese orden de prioridad) y
    tenga un valor de tipo string no vacio.
    """
    clave_real_por_lower = {}
    for key in payload.keys():
        if isinstance(key, str):
            clave_real_por_lower.setdefault(key.lower(), key)

    for campo in CAMPOS_BASE64_POSIBLES:
        clave_real = clave_real_por_lower.get(campo.lower())
        if clave_real is None:
            continue
        valor = payload[clave_real]
        if isinstance(valor, str) and valor.strip():
            return valor
    return None


def _decodificar_base64(texto: str) -> bytes:
    """Decodifica un string base64, tolerando espacios/saltos de linea
    (por si viene "wrapped"), pero rechazando cualquier otro caracter fuera
    del alfabeto base64 - asi un body que no es base64 real (texto
    arbitrario, JSON mal formado, etc.) falla aqui en vez de "decodificar"
    basura silenciosamente.
    """
    limpio = "".join(texto.split())
    return base64.b64decode(limpio, validate=True)


def _error_campo_base64(keys_recibidas: list[str]) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "error": "no_se_encontro_campo_base64_valido",
            "campos_aceptados": list(CAMPOS_BASE64_POSIBLES),
            "keys_recibidas": keys_recibidas,
        },
    )


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
    detalle = {
        "checkpoint": checkpoint,
        "duracion_audio_usada_s": duracion_audio_usada_s,
        "duracion_total_llamada_s": duracion_total_s,
        "umbral_decision_temprana": UMBRAL_DECISION_TEMPRANA,
    }
    logging.info(
        "resultado deteccion: is_synthetic=%s confidence=%s detalle=%s",
        resultado.is_synthetic,
        resultado.confidence,
        detalle,
    )
    return {
        "is_synthetic": resultado.is_synthetic,
        "confidence": resultado.confidence,
        "detalle": detalle,
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
async def detect(request: Request):
    """Contrato del campo base64 sin confirmar con los organizadores: se
    intenta, en orden, (1) JSON con alguno de CAMPOS_BASE64_POSIBLES como
    key (case-insensitive), y si el Content-Type no es application/json o
    el body no parsea como JSON valido, (2) el body crudo completo como el
    string base64 directamente (por si el WAV viene en base64 como texto
    plano, sin envolver en JSON). Ver README para el detalle.
    """
    body_bytes = await request.body()
    content_type = request.headers.get("content-type", "")
    es_json_declarado = "application/json" in content_type.lower()

    payload = None
    json_parseo_ok = False
    if body_bytes:
        try:
            candidato = json.loads(body_bytes)
            json_parseo_ok = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            candidato = None
        if isinstance(candidato, dict):
            payload = candidato

    audio_base64 = None
    keys_recibidas: list[str] = []

    if es_json_declarado and json_parseo_ok:
        if payload is not None:
            keys_recibidas = sorted(str(k) for k in payload.keys())
            audio_base64 = _buscar_campo_base64(payload)
    else:
        # Content-Type no es application/json, o el body no parseo como
        # JSON valido: se intenta el body crudo completo como el string
        # base64 directamente.
        try:
            texto_crudo = body_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            texto_crudo = ""
        if texto_crudo:
            audio_base64 = texto_crudo

    if audio_base64 is None:
        raise _error_campo_base64(keys_recibidas)

    try:
        audio_bytes = _decodificar_base64(audio_base64)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"base64 invalido: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        ruta_temporal = tmp.name

    try:
        resultado_completo = analizar_wav(ruta_temporal)
    finally:
        os.remove(ruta_temporal)

    # Contrato oficial del reto: la respuesta debe ser EXCLUSIVAMENTE
    # {"is_synthetic": bool, "confidence": float}, sin campos extra (riesgo
    # de que un validador de schema estricto del evaluador rechace la
    # respuesta completa). El detalle completo ya quedo logueado en
    # _respuesta().
    return {
        "is_synthetic": bool(resultado_completo["is_synthetic"]),
        "confidence": float(resultado_completo["confidence"]),
    }


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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "detect-altur", "ready": True}


# Montar frontend estático si existe la carpeta
_DIRECTORIO_FRONTEND = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)
if os.path.exists(_DIRECTORIO_FRONTEND):
    app.mount("/static", StaticFiles(directory=_DIRECTORIO_FRONTEND), name="static")

    @app.get("/")
    async def index():
        archivo_index = os.path.join(_DIRECTORIO_FRONTEND, "index.html")
        if os.path.exists(archivo_index):
            return FileResponse(archivo_index)
        return {"status": "ok", "service": "detect-altur"}

