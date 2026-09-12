"""calibracion/probar_semantico_openai.py — prueba aislada: reemplaza el LLM de
la Spark (SPARK_API_KEY, uncensored) por OpenAI (gpt-4o-mini) para el Paso 2 del
DetectorSemantico, SIN tocar app/deteccion/semantico.py ni ningun otro archivo
de produccion.

No es un script de calibracion completo: solo mide, sobre una muestra chica,
si el LLM de OpenAI resuelve dentro del timeout de pared y si separa human vs
synthetic al menos tan bien como el regex de honestidad.

IMPORTANTE (no existia antes de este script):
- El proyecto no tiene mecanismo de carga de .env: SPARK_API_KEY se lee
  directo de os.environ y se exporta a mano en el shell (ver docstring de
  calibrar.py). Este script agrega un parser de .env minimo, local a este
  archivo, solo para no tener que exportar OPENAI_API_KEY a mano cada vez.
- Requiere el paquete "openai" (no esta en requirements.txt ni en
  requirements-calibracion.txt): pip install openai

Uso tipico:
    pip install openai
    python calibracion/probar_semantico_openai.py
    python calibracion/probar_semantico_openai.py --muestra 20
"""
import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.semantico import DetectorSemantico, PROMPT_TEMPLATE  # noqa: E402
from app.main import cargar_canales  # noqa: E402

from calibrar import leer_manifest, muestrear_balanceado  # noqa: E402

MODELO_OPENAI = "gpt-4o-mini"
TIMEOUT_LLM_REQUEST_S = 2.0  # timeout de red al request de OpenAI (analogo a SPARK_TIMEOUT_S)
TIMEOUT_PARED_S = 3  # mismo timeout de pared que usa DetectorSemantico.analizar() hoy

MUESTRA_POR_DEFECTO = 16  # 8 human + 8 synthetic


def _cargar_env_desde_dotenv(ruta_env: Path) -> None:
    """Parser minimo de .env, local a este script (no existe un mecanismo
    equivalente en produccion: SPARK_API_KEY se exporta a mano en el shell).
    No sobreescribe variables ya presentes en el entorno.
    """
    if not ruta_env.exists():
        return
    for linea in ruta_env.read_text().splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave = clave.strip()
        valor = valor.strip().strip('"').strip("'")
        if clave:
            os.environ.setdefault(clave, valor)


_cargar_env_desde_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _clasificar_con_llm_openai(cliente, transcript: str) -> tuple[float, str | None]:
    """Mismo prompt y mismo criterio que DetectorSemantico._clasificar_con_llm
    (PROMPT_TEMPLATE de semantico.py), adaptado al formato de mensajes de la
    API de OpenAI. A diferencia de la version de produccion, NO atrapa
    excepciones aqui: se dejan subir para que el caller las clasifique en
    timeout/llm_error.
    """
    prompt = PROMPT_TEMPLATE.format(transcript=transcript)
    respuesta = cliente.chat.completions.create(
        model=MODELO_OPENAI,
        messages=[{"role": "user", "content": prompt}],
    )
    contenido = respuesta.choices[0].message.content
    parsed = DetectorSemantico._extraer_json(contenido)
    score = float(np.clip(float(parsed["score"]), 0.0, 1.0))
    return score, parsed.get("razon")


def evaluar_llamada(detector: DetectorSemantico, cliente, canal_caller, sr) -> dict:
    transcript = detector._transcribir(canal_caller, sr)
    if not transcript:
        return {"score": 0.5, "metodo": "transcript_vacio", "tiempo_llm_s": None, "razon": None}

    match = detector._filtro_regex(transcript)
    if match is not None:
        return {"score": 0.15, "metodo": "regex", "tiempo_llm_s": None, "razon": match}

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_clasificar_con_llm_openai, cliente, transcript)
    t0 = time.time()
    try:
        score, razon = future.result(timeout=TIMEOUT_PARED_S)
        return {"score": score, "metodo": "llm", "tiempo_llm_s": time.time() - t0, "razon": razon}
    except concurrent.futures.TimeoutError:
        return {"score": 0.5, "metodo": "timeout", "tiempo_llm_s": time.time() - t0, "razon": None}
    except Exception as exc:  # error de red/parseo del LLM, no timeout de pared
        return {"score": 0.5, "metodo": "llm_error", "tiempo_llm_s": time.time() - t0, "razon": repr(exc)}
    finally:
        executor.shutdown(wait=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="/home/andres/hackmty26/manifest.csv")
    parser.add_argument("--audio-dir", default="/home/andres/hackmty26/audio")
    parser.add_argument("--muestra", type=int, default=MUESTRA_POR_DEFECTO, help="tamano de muestra balanceada por clase")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY no esta seteada (revisa detect-altur/.env)")

    import openai  # import tardio: falla con mensaje claro si no esta instalado

    cliente = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=TIMEOUT_LLM_REQUEST_S)
    detector = DetectorSemantico()

    filas_train = leer_manifest(Path(args.manifest), split="train")
    muestra = muestrear_balanceado(filas_train, args.muestra, args.semilla)
    print(f"Muestra: {len(muestra)} llamadas balanceadas (semilla={args.semilla})")

    resultados = []
    for i, fila in enumerate(muestra, start=1):
        anon_id = fila["anon_id"]
        ruta_audio = Path(args.audio_dir) / f"{anon_id}.wav"
        if not ruta_audio.exists():
            print(f"[{i}/{len(muestra)}] {anon_id}: audio no encontrado, se omite")
            continue

        canal_caller, _canal_callee, sr = cargar_canales(str(ruta_audio))
        r = evaluar_llamada(detector, cliente, canal_caller, sr)
        r["anon_id"] = anon_id
        r["label"] = fila["label"]
        resultados.append(r)

        tiempo_str = f" tiempo_llm={r['tiempo_llm_s']:.2f}s" if r["tiempo_llm_s"] is not None else ""
        print(
            f"[{i}/{len(muestra)}] {anon_id} label={r['label']} score={r['score']:.3f} "
            f"metodo={r['metodo']}{tiempo_str}"
        )

    if not resultados:
        print("Sin resultados utilizables.")
        return

    n_timeout = sum(1 for r in resultados if r["metodo"] == "timeout")
    n_regex = sum(1 for r in resultados if r["metodo"] == "regex")
    n_llm_real = sum(1 for r in resultados if r["metodo"] == "llm")
    n_llm_error = sum(1 for r in resultados if r["metodo"] == "llm_error")
    n_vacio = sum(1 for r in resultados if r["metodo"] == "transcript_vacio")

    print("\n=== Resumen ===")
    print(f"Total evaluadas: {len(resultados)}")
    print(f"Resueltas por regex (paso 1): {n_regex}")
    print(f"Resueltas por LLM con score real: {n_llm_real}")
    print(f"Timeout de pared ({TIMEOUT_PARED_S}s): {n_timeout}")
    print(f"Error de LLM (no timeout, ej. fallo de red/parseo): {n_llm_error}")
    print(f"Transcript vacio: {n_vacio}")

    # Misma formula que calibrar.py Paso 3 (reportar_separacion_por_senal):
    # |diff de medias| sobre TODOS los scores obtenidos (regex + LLM + fallbacks).
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in resultados])
    scores = np.array([r["score"] for r in resultados])
    h, s = scores[y == 0], scores[y == 1]
    print(
        f"\nsemantico (con OpenAI en paso 2): humano mean={h.mean():.3f} std={h.std():.3f} | "
        f"sintetico mean={s.mean():.3f} std={s.std():.3f} | "
        f"|diff medias|={abs(s.mean() - h.mean()):.3f}"
    )


if __name__ == "__main__":
    main()
