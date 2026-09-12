"""validar_val_http.py — valida el endpoint HTTP real POST /detect (no las
funciones de deteccion llamadas en directo) contra las 71 llamadas reales de
split="val", midiendo accuracy/matriz de confusion/EER Y latencia real por
request (evidencia real para el criterio de Latency del reto, n=71 en vez de
1-2 muestras sueltas).

Requiere que el servidor ya este corriendo en otra terminal:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Diferencia clave con validar_val.py: ese script llama
DetectorComportamiento.analizar() + Fusion.combinar() directamente sobre el
audio COMPLETO de cada llamada, sin pasar por decidir_secuencial() en
app/main.py — nunca ejercita la logica de checkpoints (25s/40s,
UMBRAL_DECISION_TEMPRANA=0.75) que SI corre en produccion real via /detect.
Este script si pasa por esa ruta completa (HTTP -> decidir_secuencial ->
checkpoints -> Fusion), asi que si accuracy/EER no coinciden entre ambos
scripts, la sospecha inmediata es esa logica de checkpoints — ver Paso 4.

No modifica fusion.py, main.py ni ningun peso — solo mide, vía HTTP real.

Guarda resultados crudos incrementalmente en --salida (jsonl) y puede
reanudarse si se interrumpe: al volver a correr, salta las llamadas ya
procesadas en ese archivo (usa --reiniciar para ignorarlas y empezar de cero).

Uso tipico:
    python validar_val_http.py
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import accuracy_score, confusion_matrix

# Este script vive en calibracion/, pero necesita importar validar_val.py
# (mismo directorio) para reutilizar leer_manifest y calcular_eer sin
# duplicar esa logica.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validar_val import calcular_eer, leer_manifest

URL_POR_DEFECTO = "http://localhost:8000/detect"
TIMEOUT_POR_DEFECTO_S = 60.0
SALIDA_POR_DEFECTO = Path(__file__).resolve().parent / "validacion_val_http_resultados.jsonl"

# Numeros de referencia de validar_val.py (llamada DIRECTA a
# DetectorComportamiento.analizar() + Fusion.combinar() sobre el audio
# completo de las mismas 71 llamadas de val, sin pasar por
# decidir_secuencial() ni por los checkpoints de app/main.py). Se usan aqui
# solo para comparar al final — ver Paso 4. Si corres validar_val.py de
# nuevo y esos numeros cambian, actualiza estas dos constantes.
ACCURACY_VALIDAR_DIRECTO = 0.732
EER_VALIDAR_DIRECTO = 0.267


def cargar_registros_previos(ruta_salida: Path) -> list[dict]:
    if not ruta_salida.exists():
        return []
    registros = []
    for linea in ruta_salida.read_text().splitlines():
        linea = linea.strip()
        if linea:
            registros.append(json.loads(linea))
    return registros


def llamar_detect(url: str, ruta_audio: Path, timeout_s: float) -> tuple[dict | None, float, str | None]:
    """POST real a /detect con el WAV codificado en base64 (libreria base64
    de Python, nunca shell/curl). Devuelve (respuesta, latencia_s, error):
    si error no es None, respuesta es None y la llamada se debe contar como
    fallo, no como una prediccion.
    """
    audio_bytes = ruta_audio.read_bytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    payload = {"audio_base64": audio_base64}

    t0 = time.perf_counter()
    try:
        respuesta_http = requests.post(url, json=payload, timeout=timeout_s)
        latencia_s = time.perf_counter() - t0
        respuesta_http.raise_for_status()
    except requests.exceptions.RequestException as exc:
        latencia_s = time.perf_counter() - t0
        return None, latencia_s, f"{type(exc).__name__}: {exc}"

    try:
        cuerpo = respuesta_http.json()
        is_synthetic = bool(cuerpo["is_synthetic"])
        confidence = float(cuerpo["confidence"])
    except (ValueError, KeyError, TypeError) as exc:
        return None, latencia_s, f"respuesta invalida ({exc!r}): {respuesta_http.text[:200]!r}"

    return {"is_synthetic": is_synthetic, "confidence": confidence}, latencia_s, None


def recolectar_datos(
    filas: list[dict],
    audio_dir: Path,
    url: str,
    timeout_s: float,
    ruta_salida: Path,
    reiniciar: bool,
) -> list[dict]:
    registros = [] if reiniciar else cargar_registros_previos(ruta_salida)
    ya_procesadas = {r["anon_id"] for r in registros}
    if ya_procesadas:
        print(f"Reanudando: {len(ya_procesadas)} llamadas ya procesadas en {ruta_salida}, se saltan.")

    pendientes = [f for f in filas if f["anon_id"] not in ya_procesadas]
    if not pendientes:
        return registros

    modo_escritura = "a" if (registros and not reiniciar) else "w"
    archivo_salida = open(ruta_salida, modo_escritura)

    for i, fila in enumerate(pendientes, start=1):
        anon_id = fila["anon_id"]
        ruta_audio = audio_dir / f"{anon_id}.wav"
        if not ruta_audio.exists():
            print(f"  [{i}/{len(pendientes)}] {anon_id}: audio no encontrado en {ruta_audio}, se omite")
            continue

        respuesta, latencia_s, error = llamar_detect(url, ruta_audio, timeout_s)
        registro = {
            "anon_id": anon_id,
            "label": fila["label"],
            "latencia_s": latencia_s,
            "respuesta": respuesta,
            "error": error,
        }
        registros.append(registro)
        archivo_salida.write(json.dumps(registro) + "\n")
        archivo_salida.flush()

        if error is not None:
            print(f"  [{i}/{len(pendientes)}] {anon_id} ({fila['label']}): FALLO ({latencia_s:.2f}s) — {error}")
        else:
            print(
                f"  [{i}/{len(pendientes)}] {anon_id} ({fila['label']}): "
                f"is_synthetic={respuesta['is_synthetic']} confidence={respuesta['confidence']:.3f} "
                f"({latencia_s:.2f}s)"
            )

    archivo_salida.close()
    return registros


def reportar_fallos(registros: list[dict]) -> list[dict]:
    fallos = [r for r in registros if r["error"] is not None]
    print("\n=== Paso 1: fallos de red/HTTP ===")
    if not fallos:
        print(f"0 fallos — las {len(registros)} llamadas respondieron 200 con un body valido.")
    else:
        print(f"{len(fallos)}/{len(registros)} llamadas fallaron:")
        for r in fallos:
            print(f"  {r['anon_id']} ({r['label']}): {r['error']}")
    return fallos


def reportar_accuracy_eer(registros_ok: list[dict]) -> tuple[float, float]:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros_ok])
    pred = np.array([1 if r["respuesta"]["is_synthetic"] else 0 for r in registros_ok])
    conf = np.array([r["respuesta"]["confidence"] for r in registros_ok])

    accuracy = float(accuracy_score(y, pred))
    eer = calcular_eer(y, conf)

    print(f"\n=== Paso 2: accuracy, matriz de confusion y EER vía HTTP real (n={len(registros_ok)}) ===")
    print("Accuracy:", accuracy)
    print("Matriz de confusion [[TN,FP],[FN,TP]]:\n", confusion_matrix(y, pred))
    print("EER:", eer)
    return accuracy, eer


def reportar_latencia(registros_ok: list[dict]) -> None:
    latencias = np.array([r["latencia_s"] for r in registros_ok])
    print(f"\n=== Paso 3: latencia real por request (n={len(latencias)}, /detect end-to-end) ===")
    print(f"media:       {latencias.mean():.3f}s")
    print(f"mediana:     {np.median(latencias):.3f}s")
    print(f"minimo:      {latencias.min():.3f}s")
    print(f"maximo:      {latencias.max():.3f}s")
    print(f"percentil95: {np.percentile(latencias, 95):.3f}s")


def reportar_comparacion_directo(accuracy_http: float, eer_http: float) -> None:
    print("\n=== Paso 4: HTTP real vs. llamada directa (validar_val.py) ===")
    print(
        f"  Accuracy: directo={ACCURACY_VALIDAR_DIRECTO:.3f}  http={accuracy_http:.3f}  "
        f"diff={accuracy_http - ACCURACY_VALIDAR_DIRECTO:+.3f}"
    )
    print(
        f"  EER:      directo={EER_VALIDAR_DIRECTO:.3f}  http={eer_http:.3f}  "
        f"diff={eer_http - EER_VALIDAR_DIRECTO:+.3f}"
    )
    if abs(accuracy_http - ACCURACY_VALIDAR_DIRECTO) < 1e-9:
        print(
            "  Coinciden exactamente: la logica de checkpoints/decision secuencial de "
            "main.py no cambio el resultado sobre val."
        )
    else:
        print(
            "  NO coinciden — senal de que decidir_secuencial() (checkpoints 25s/40s + "
            "UMBRAL_DECISION_TEMPRANA=0.75 en app/main.py) esta produciendo, en al menos "
            "algunas llamadas, una decision distinta a la de evaluar la llamada completa "
            "directamente. validar_val.py nunca ejercita esa logica de checkpoints; este "
            "script si, porque pasa por /detect real."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default="/home/andres/hackmty26/manifest.csv")
    parser.add_argument("--audio-dir", default="/home/andres/hackmty26/audio")
    parser.add_argument("--url", default=URL_POR_DEFECTO)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_POR_DEFECTO_S)
    parser.add_argument("--salida", default=str(SALIDA_POR_DEFECTO))
    parser.add_argument(
        "--reiniciar", action="store_true", help="ignora --salida existente y empieza de cero"
    )
    args = parser.parse_args()

    filas_val = leer_manifest(Path(args.manifest), split="val")
    print(f"Llamadas en split=val: {len(filas_val)}")
    print(f"Endpoint: {args.url} (timeout={args.timeout:.0f}s por request)")

    registros = recolectar_datos(
        filas_val, Path(args.audio_dir), args.url, args.timeout, Path(args.salida), args.reiniciar
    )
    print(f"\nTotal de llamadas con respuesta registrada: {len(registros)}/{len(filas_val)}")

    reportar_fallos(registros)
    registros_ok = [r for r in registros if r["error"] is None]

    if len(registros_ok) < 10:
        print("Muy pocos registros utilizables para medir accuracy/EER/latencia de forma confiable. Saliendo.")
        return

    accuracy_http, eer_http = reportar_accuracy_eer(registros_ok)
    reportar_latencia(registros_ok)
    reportar_comparacion_directo(accuracy_http, eer_http)


if __name__ == "__main__":
    main()
