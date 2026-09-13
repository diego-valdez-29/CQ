"""smoke_test_final.py — smoke test contra el endpoint HTTP real (POST
/detect) sobre un conjunto fijo de llamadas de control cuyo resultado ya
fue validado manualmente (ver conversacion/registro de calibracion de esta
noche): compara is_synthetic y confidence contra el valor esperado, con
tolerancia 0.001 en confidence.

No es una validacion estadistica (para eso esta validar_val_http.py sobre
las 71 llamadas de val) — es un smoke test rapido de regresion: si alguno
de estos 3 numeros cambia, algo en el pipeline (pesos, umbral, checkpoints,
version de un detector) se movio respecto a lo que se valido esta noche.

Requiere que el servidor ya este corriendo en otra terminal:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Uso tipico:
    python smoke_test_final.py
"""
import argparse
import base64
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

URL_POR_DEFECTO = "http://localhost:8000/detect"
TIMEOUT_POR_DEFECTO_S = 60.0
TOLERANCIA_CONFIDENCE = 0.001

AUDIO_DIR_POR_DEFECTO = Path("/home/andres/Downloads/audio")


@dataclass(frozen=True)
class CasoControl:
    anon_id: str
    label: str
    is_synthetic_esperado: bool
    confidence_esperada: float


CASOS = (
    CasoControl("call_e69e8ba551e4", "human", False, 0.572),
    CasoControl("call_dcdbcc7d70e4", "synthetic", True, 0.617),
    CasoControl("call_bc2c7c34acea", "human", False, 0.535),
)


def llamar_detect(url: str, ruta_audio: Path, timeout_s: float) -> tuple[dict | None, str | None]:
    """POST real a /detect con el WAV codificado en base64 (libreria base64
    de Python, nunca shell/curl). Devuelve (respuesta, error): si error no
    es None, respuesta es None.
    """
    audio_bytes = ruta_audio.read_bytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    payload = {"audio_base64": audio_base64}

    try:
        respuesta_http = requests.post(url, json=payload, timeout=timeout_s)
        respuesta_http.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        cuerpo = respuesta_http.json()
        is_synthetic = bool(cuerpo["is_synthetic"])
        confidence = float(cuerpo["confidence"])
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"respuesta invalida ({exc!r}): {respuesta_http.text[:200]!r}"

    return {"is_synthetic": is_synthetic, "confidence": confidence}, None


def evaluar_caso(caso: CasoControl, audio_dir: Path, url: str, timeout_s: float) -> bool:
    ruta_audio = audio_dir / f"{caso.anon_id}.wav"
    if not ruta_audio.exists():
        print(f"FAIL {caso.anon_id} ({caso.label}): audio no encontrado en {ruta_audio}")
        return False

    respuesta, error = llamar_detect(url, ruta_audio, timeout_s)
    if error is not None:
        print(f"FAIL {caso.anon_id} ({caso.label}): {error}")
        return False

    is_synthetic_ok = respuesta["is_synthetic"] == caso.is_synthetic_esperado
    diff_confidence = abs(respuesta["confidence"] - caso.confidence_esperada)
    confidence_ok = diff_confidence <= TOLERANCIA_CONFIDENCE
    ok = is_synthetic_ok and confidence_ok

    estado = "PASS" if ok else "FAIL"
    print(
        f"{estado} {caso.anon_id} ({caso.label}): "
        f"is_synthetic={respuesta['is_synthetic']} (esperado {caso.is_synthetic_esperado}) "
        f"confidence={respuesta['confidence']:.4f} "
        f"(esperado {caso.confidence_esperada:.4f}, diff={diff_confidence:.4f}, tol={TOLERANCIA_CONFIDENCE})"
    )
    if not is_synthetic_ok:
        print(f"     -> is_synthetic no coincide")
    if not confidence_ok:
        print(f"     -> confidence fuera de tolerancia")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR_POR_DEFECTO))
    parser.add_argument("--url", default=URL_POR_DEFECTO)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_POR_DEFECTO_S)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    print(f"Endpoint: {args.url} (timeout={args.timeout:.0f}s por request)")
    print(f"Audio dir: {audio_dir}\n")

    resultados = [evaluar_caso(caso, audio_dir, args.url, args.timeout) for caso in CASOS]

    n_ok = sum(resultados)
    n_total = len(resultados)
    print(f"\n=== Resumen: {n_ok}/{n_total} PASS ===")

    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
