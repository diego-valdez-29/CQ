"""calibrar_comportamiento_crudo.py — extrae std_recuperacion y
tiempo_primera_habla CRUDOS (antes de normalizar en
DetectorComportamiento._score_heuristico) para las mismas 30 llamadas que
calibrar_acustico_crudo.py (--muestra 30, --semilla 0), sin repetir Whisper
ni Spark: usa la misma seleccion balanceada sobre el manifest y solo corre
DetectorComportamiento (VAD local con silero-vad).

tiempo_primera_habla es el timestamp (segundos) del inicio del primer
segmento de habla detectado en canal_caller — no requiere VAD adicional,
ya sale de _timestamps_habla.

No se ejecuta automaticamente. Correr manualmente con:
    python calibrar_comportamiento_crudo.py
"""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

# Este script vive en calibracion/, pero "app" esta un nivel arriba (raiz del
# proyecto). Python solo agrega el directorio del propio script a sys.path,
# no el cwd, asi que sin esto el import de abajo falla al correrlo desde
# fuera de calibracion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.main import cargar_canales

MANIFEST = Path("/home/andres/hackmty26/manifest.csv")
AUDIO_DIR = Path("/home/andres/hackmty26/audio")
SALIDA = Path(__file__).resolve().parent / "calibracion_comportamiento_crudo.json"

MUESTRA = 30
SEMILLA = 0

# Mismo umbral que DetectorComportamiento.analizar(): con menos de 2
# recuperaciones no hay std confiable y el detector cae a score=0.5.
MINIMO_RECUPERACIONES = 2


def leer_manifest(ruta_manifest: Path, split: str) -> list[dict]:
    with open(ruta_manifest, newline="") as f:
        return [fila for fila in csv.DictReader(f) if fila["split"] == split]


def muestrear_balanceado(filas: list[dict], n: int, semilla: int) -> list[dict]:
    rng = random.Random(semilla)
    humanos = [f for f in filas if f["label"] == "human"]
    sinteticos = [f for f in filas if f["label"] == "synthetic"]
    rng.shuffle(humanos)
    rng.shuffle(sinteticos)
    n_por_clase = n // 2
    return humanos[:n_por_clase] + sinteticos[:n_por_clase]


def main() -> None:
    filas_train = leer_manifest(MANIFEST, split="train")
    muestra = muestrear_balanceado(filas_train, MUESTRA, SEMILLA)
    print(f"Llamadas en split=train: {len(filas_train)} — muestra: {len(muestra)}")

    detector = DetectorComportamiento()
    resultados = []

    for fila in muestra:
        anon_id = fila["anon_id"]
        ruta = AUDIO_DIR / f"{anon_id}.wav"
        try:
            canal_caller, canal_callee, sr = cargar_canales(str(ruta))

            audio_caller = detector._resamplear(canal_caller, sr)
            audio_callee = detector._resamplear(canal_callee, sr)

            habla_caller = detector._timestamps_habla(audio_caller)
            habla_callee = detector._timestamps_habla(audio_callee)

            eventos = detector._detectar_eventos(habla_callee, habla_caller)
            recuperaciones = detector._tiempos_recuperacion(eventos, habla_caller)
            tiempo_primera_habla = (
                float(habla_caller[0]["start"]) if habla_caller else None
            )
        except Exception as exc:
            resultados.append({
                "anon_id": anon_id,
                "label": fila["label"],
                "error": str(exc),
            })
            print(f"{anon_id} ({fila['label']}): ERROR {exc!r}")
            continue

        eventos_insuficientes = len(recuperaciones) < MINIMO_RECUPERACIONES
        std_recuperacion = float(np.std(recuperaciones)) if recuperaciones else None
        media_recuperacion = float(np.mean(recuperaciones)) if recuperaciones else None

        resultado = {
            "anon_id": anon_id,
            "label": fila["label"],
            "numero_eventos_detectados": len(eventos),
            "numero_recuperaciones": len(recuperaciones),
            "std_recuperacion": std_recuperacion,
            "media_recuperacion": media_recuperacion,
            "tiempo_primera_habla": tiempo_primera_habla,
            "eventos_insuficientes": eventos_insuficientes,
        }
        resultados.append(resultado)
        print(
            f"{anon_id} ({fila['label']}): eventos={len(eventos)} "
            f"recuperaciones={len(recuperaciones)} "
            f"std_recuperacion={std_recuperacion} "
            f"tiempo_primera_habla={tiempo_primera_habla} "
            f"eventos_insuficientes={eventos_insuficientes}"
        )

    SALIDA.write_text(json.dumps(resultados, indent=2))
    print(f"\nGuardado en {SALIDA} ({len(resultados)} llamadas)")

    procesadas = [r for r in resultados if "error" not in r]
    for clase in ("human", "synthetic"):
        de_la_clase = [r for r in procesadas if r["label"] == clase]
        insuficientes = [r for r in de_la_clase if r["eventos_insuficientes"]]
        print(
            f"eventos_insuficientes en {clase}: {len(insuficientes)}/{len(de_la_clase)}"
        )
    con_error = [r for r in resultados if "error" in r]
    if con_error:
        print(f"({len(con_error)} llamadas fallaron antes de llegar a VAD, ver 'error' en {SALIDA})")

    # --- Verificacion del score final de produccion (solo tiempo_primera_habla,
    # ver DetectorComportamiento._score_heuristico) ---
    con_tiempo = [r for r in procesadas if r["tiempo_primera_habla"] is not None]
    if len(con_tiempo) < 2:
        print("\nNo hay suficientes llamadas con tiempo_primera_habla valido "
              "para verificar el score final.")
        return

    for r in con_tiempo:
        r["score_final"] = detector._score_heuristico(r["tiempo_primera_habla"])

    print(
        f"\nPoblacion (n={len(con_tiempo)}): "
        f"media_tiempo_primera_habla={detector._media_tiempo_primera_habla:.5f} "
        f"std_tiempo_primera_habla={detector._std_tiempo_primera_habla:.5f}"
    )

    print("\nScore final por clase (threshold=0.5 -> sintetico):")
    medias_por_clase = {}
    correctos = 0
    for clase in ("human", "synthetic"):
        de_la_clase = [r for r in con_tiempo if r["label"] == clase]
        scores = [r["score_final"] for r in de_la_clase]
        media_clase = float(np.mean(scores))
        medias_por_clase[clase] = media_clase
        prediccion_correcta = (
            [s < 0.5 for s in scores] if clase == "human" else [s >= 0.5 for s in scores]
        )
        correctos += sum(prediccion_correcta)
        print(f"  {clase}: n={len(de_la_clase)} media_score={media_clase:.5f}")

    diff = abs(medias_por_clase["human"] - medias_por_clase["synthetic"])
    accuracy = correctos / len(con_tiempo)
    print(f"\n|diff de medias| (score final) = {diff:.5f}")
    print(f"accuracy (threshold=0.5) = {accuracy:.4f} ({correctos}/{len(con_tiempo)})")


if __name__ == "__main__":
    main()
