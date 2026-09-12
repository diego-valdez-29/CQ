"""calibrar_acustico_crudo.py — extrae varianza_mfcc y regularidad_espectral
CRUDOS (antes del sigmoid de DetectorAcustico._score_heuristico) para las 30
llamadas que ya esta usando la corrida en curso de calibrar.py (--muestra 30,
--semilla 0 por defecto), sin repetir Whisper ni Spark: usa la misma
seleccion balanceada sobre el manifest y solo corre DetectorAcustico.

No se ejecuta automaticamente. Correr manualmente con:
    python calibrar_acustico_crudo.py
"""
import csv
import json
import random
import sys
from pathlib import Path

# Este script vive en calibracion/, pero "app" esta un nivel arriba (raiz del
# proyecto). Python solo agrega el directorio del propio script a sys.path,
# no el cwd, asi que sin esto el import de abajo falla al correrlo desde
# fuera de calibracion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.acustico import DetectorAcustico
from app.main import cargar_canales

MANIFEST = Path("/home/andres/hackmty26/manifest.csv")
AUDIO_DIR = Path("/home/andres/hackmty26/audio")
SALIDA = Path(__file__).resolve().parent / "calibracion_acustico_crudo.json"

MUESTRA = 30
SEMILLA = 0


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

    detector = DetectorAcustico()
    resultados = []

    for fila in muestra:
        anon_id = fila["anon_id"]
        ruta = AUDIO_DIR / f"{anon_id}.wav"
        try:
            canal_caller, canal_callee, sr = cargar_canales(str(ruta))
            audio, _ = detector._preprocesar(canal_caller, sr)
            if not detector._verificar_condiciones_analisis(audio):
                resultados.append({
                    "anon_id": anon_id,
                    "label": fila["label"],
                    "error": "audio_insuficiente",
                })
                continue
            features = detector._extraer_caracteristicas(audio)
        except Exception as exc:
            resultados.append({
                "anon_id": anon_id,
                "label": fila["label"],
                "error": str(exc),
            })
            continue

        resultados.append({
            "anon_id": anon_id,
            "label": fila["label"],
            "varianza_mfcc": features["varianza_mfcc"],
            "regularidad_espectral": features["regularidad_espectral"],
        })
        print(f"{anon_id} ({fila['label']}): {features}")

    SALIDA.write_text(json.dumps(resultados, indent=2))
    print(f"\nGuardado en {SALIDA} ({len(resultados)} llamadas)")


if __name__ == "__main__":
    main()
