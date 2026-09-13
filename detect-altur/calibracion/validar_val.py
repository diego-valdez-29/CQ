"""validar_val.py — evalua (NO recalibra) el detector de produccion sobre las
71 llamadas de split="val" del manifest, nunca usadas en ninguna calibracion
anterior (todas las calibraciones de calibrar.py / calibrar_*.py corrieron
sobre split="train").

Corre EXACTAMENTE el mismo pipeline que /detect en produccion (app/main.py):
solo DetectorComportamiento, fusionado con Fusion (app/deteccion/fusion.py)
usando los pesos y umbral ya fijados en produccion:

    pesos  = {"acustico": 0.0, "comportamiento": 1.0, "semantico": 0.0}
    umbral = 0.56

No modifica fusion.py, main.py ni ningun peso — solo mide accuracy, matriz de
confusion, EER, Brier score y un reliability diagram sobre val, y compara
contra los numeros ya reportados en README.md para train (accuracy=0.833,
EER=0.183) para detectar sobreajuste al split de calibracion.

Guarda resultados crudos incrementalmente en --salida (jsonl) y puede
reanudarse si se interrumpe: al volver a correr, salta las llamadas ya
procesadas en ese archivo (usa --reiniciar para ignorarlas y empezar de cero).

Uso tipico:
    python validar_val.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

# Este script vive en calibracion/, pero "app" esta un nivel arriba (raiz del
# proyecto). Python solo agrega el directorio del propio script a sys.path,
# no el cwd, asi que sin esto el import de abajo falla al correrlo desde
# fuera de calibracion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.deteccion.fusion import Fusion
from app.dominio.modelos import SenalScore
from app.main import cargar_canales

# Mismos pesos y umbral que app/main.py en produccion (ver README.md,
# seccion "Estado honesto de calibracion"). No se tocan aqui.
PESOS_PRODUCCION = {"acustico": 0.0, "comportamiento": 1.0, "semantico": 0.0}
UMBRAL_PRODUCCION = 0.56

# Numeros ya reportados en README.md, calibrados sobre n=60 de split=train.
# Sirven de referencia para detectar caida (overfitting) sobre val.
ACCURACY_TRAIN = 0.833
EER_TRAIN = 0.183

SALIDA_POR_DEFECTO = Path(__file__).resolve().parent / "validacion_val_resultados.jsonl"
BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

UMBRAL_SWEEP_INICIO = 0.30
UMBRAL_SWEEP_FIN = 0.70
UMBRAL_SWEEP_PASO = 0.02


def leer_manifest(ruta_manifest: Path, split: str) -> list[dict]:
    with open(ruta_manifest, newline="") as f:
        return [fila for fila in csv.DictReader(f) if fila["split"] == split]


def calcular_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def cargar_registros_previos(ruta_salida: Path) -> list[dict]:
    if not ruta_salida.exists():
        return []
    registros = []
    for linea in ruta_salida.read_text().splitlines():
        linea = linea.strip()
        if linea:
            registros.append(json.loads(linea))
    return registros


def recolectar_datos(filas: list[dict], audio_dir: Path, ruta_salida: Path, reiniciar: bool) -> list[dict]:
    registros = [] if reiniciar else cargar_registros_previos(ruta_salida)
    ya_procesadas = {r["anon_id"] for r in registros}
    if ya_procesadas:
        print(f"Reanudando: {len(ya_procesadas)} llamadas ya procesadas en {ruta_salida}, se saltan.")

    pendientes = [f for f in filas if f["anon_id"] not in ya_procesadas]
    if not pendientes:
        return registros

    detector = DetectorComportamiento()
    modo_escritura = "a" if (registros and not reiniciar) else "w"
    archivo_salida = open(ruta_salida, modo_escritura)

    for i, fila in enumerate(pendientes, start=1):
        anon_id = fila["anon_id"]
        ruta_audio = audio_dir / f"{anon_id}.wav"
        if not ruta_audio.exists():
            print(f"  [{i}/{len(pendientes)}] {anon_id}: audio no encontrado en {ruta_audio}, se omite")
            continue

        try:
            canal_caller, canal_callee, sr = cargar_canales(str(ruta_audio))
            senal = detector.analizar(canal_caller, canal_callee, sr)
        except Exception as exc:
            print(f"  [{i}/{len(pendientes)}] {anon_id}: FALLO ({exc!r}), se omite")
            continue

        registro = {
            "anon_id": anon_id,
            "label": fila["label"],
            "score_comportamiento": senal.score,
        }
        registros.append(registro)
        archivo_salida.write(json.dumps(registro) + "\n")
        archivo_salida.flush()
        print(f"  [{i}/{len(pendientes)}] {anon_id} ({fila['label']}): score={senal.score:.3f}")

    archivo_salida.close()
    return registros


def fusionar(score_comportamiento: float):
    senales = [SenalScore(nombre="comportamiento", score=score_comportamiento, detalle={})]
    return Fusion(pesos=PESOS_PRODUCCION, umbral=UMBRAL_PRODUCCION).combinar(senales)


def reportar_accuracy_eer(registros: list[dict]) -> None:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])
    resultados = [fusionar(r["score_comportamiento"]) for r in registros]
    pred = np.array([r.is_synthetic for r in resultados], dtype=int)
    conf = np.array([r.confidence for r in resultados])

    accuracy = accuracy_score(y, pred)
    eer = calcular_eer(y, conf)

    print(f"\n=== Paso 1: accuracy, matriz de confusion y EER sobre split=val (n={len(registros)}) ===")
    print(f"pesos={PESOS_PRODUCCION} umbral={UMBRAL_PRODUCCION}")
    print("Accuracy:", accuracy)
    print("Matriz de confusion [[TN,FP],[FN,TP]]:\n", confusion_matrix(y, pred))
    print("EER:", eer)

    print(f"\nComparacion contra train (n=60, calibracion final, ver README.md):")
    print(f"  Accuracy: train={ACCURACY_TRAIN:.3f}  val={accuracy:.3f}  diff={accuracy - ACCURACY_TRAIN:+.3f}")
    print(f"  EER:      train={EER_TRAIN:.3f}  val={eer:.3f}  diff={eer - EER_TRAIN:+.3f}")
    caida_accuracy = ACCURACY_TRAIN - accuracy
    subida_eer = eer - EER_TRAIN
    if caida_accuracy > 0.10 or subida_eer > 0.10:
        print(
            "  AVISO: caida/degradacion > 0.10 respecto a train — posible senal de "
            "overfitting al split de calibracion."
        )
    else:
        print("  Sin caida significativa (> 0.10) respecto a train.")


def reportar_brier(registros: list[dict]) -> float:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros], dtype=float)
    conf = np.array([fusionar(r["score_comportamiento"]).confidence for r in registros])
    brier = float(np.mean((conf - y) ** 2))
    print(f"\n=== Paso 2: Brier score sobre split=val ===")
    print(f"Brier score: {brier:.3f}  (0 = perfectamente calibrado, 0.25 = tan bueno como predecir 0.5 siempre)")
    return brier


def reportar_reliability_diagram(registros: list[dict]) -> None:
    confs = np.array([fusionar(r["score_comportamiento"]).confidence for r in registros])
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])

    print(f"\n=== Paso 3: reliability diagram (5 buckets por confidence) ===")
    for lo, hi in BUCKETS:
        if hi == 1.0:
            en_bucket = (confs >= lo) & (confs <= hi)
        else:
            en_bucket = (confs >= lo) & (confs < hi)
        n = int(en_bucket.sum())
        if n == 0:
            print(f"  [{lo:.1f}-{hi:.1f}): n=0 llamadas")
            continue
        frac_sintetico = float(y[en_bucket].mean())
        punto_medio = (lo + hi) / 2
        print(
            f"  [{lo:.1f}-{hi:.1f}): n={n:2d} llamadas | "
            f"fraccion real sintetica={frac_sintetico:.3f} "
            f"(esperado ~{punto_medio:.2f} si bien calibrado)"
        )


def reportar_barrido_umbral(registros: list[dict]) -> None:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])
    conf = np.array([fusionar(r["score_comportamiento"]).confidence for r in registros])
    # El EER es una propiedad de la distribucion de scores (punto donde FPR=FNR
    # en la curva ROC), no depende del umbral de decision elegido — se reporta
    # una sola vez, no por cada umbral del barrido.
    eer = calcular_eer(y, conf)

    umbrales = np.round(np.arange(UMBRAL_SWEEP_INICIO, UMBRAL_SWEEP_FIN + 1e-9, UMBRAL_SWEEP_PASO), 2)
    accuracies = np.array([accuracy_score(y, conf >= u) for u in umbrales])

    print(
        f"\n=== Paso 4: barrido de umbral ({UMBRAL_SWEEP_INICIO:.2f}-{UMBRAL_SWEEP_FIN:.2f}, "
        f"paso {UMBRAL_SWEEP_PASO:.2f}) sobre split=val ==="
    )
    print(f"EER={eer:.3f} (no depende del umbral elegido, es propiedad de la distribucion de scores)")
    for u, acc in zip(umbrales, accuracies):
        marca = "  <- umbral actual en produccion" if abs(u - UMBRAL_PRODUCCION) < 1e-9 else ""
        print(f"  umbral={u:.2f}  accuracy={acc:.3f}{marca}")

    max_accuracy = float(accuracies.max())
    umbrales_optimos = umbrales[np.isclose(accuracies, max_accuracy)]
    umbral_optimo = float(umbrales_optimos[np.argmin(np.abs(umbrales_optimos - UMBRAL_PRODUCCION))])
    accuracy_actual = float(accuracy_score(y, conf >= UMBRAL_PRODUCCION))

    print(
        f"\nUmbral(es) que maximizan accuracy en val: "
        f"{[f'{u:.2f}' for u in umbrales_optimos]} (accuracy={max_accuracy:.3f})"
    )
    print("Comparacion lado a lado:")
    print(f"  umbral actual (produccion, Fusion.umbral en main.py) = {UMBRAL_PRODUCCION:.2f}  ->  accuracy={accuracy_actual:.3f}")
    print(f"  umbral optimo en val                                  = {umbral_optimo:.2f}  ->  accuracy={max_accuracy:.3f}")

    diff = max_accuracy - accuracy_actual
    if abs(UMBRAL_PRODUCCION - umbral_optimo) < 1e-9:
        print("  El umbral actual YA es el optimo en val — no hay motivo para cambiar Fusion.umbral en main.py.")
    elif diff < 0.02:
        print(
            f"  El optimo en val ({umbral_optimo:.2f}) mejora accuracy solo {diff:+.3f} sobre el umbral actual "
            "— diferencia marginal sobre n=71, probablemente no justifica tocar Fusion.umbral en main.py "
            "(riesgo de ajustar al ruido de esta muestra)."
        )
    else:
        print(
            f"  El optimo en val ({umbral_optimo:.2f}) mejora accuracy {diff:+.3f} sobre el umbral actual "
            f"({UMBRAL_PRODUCCION:.2f}) — considerar actualizar Fusion.umbral en app/main.py, pero validar "
            "primero con mas datos antes de tocar produccion."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default="/home/andres/hackmty26/manifest.csv")
    parser.add_argument("--audio-dir", default="/home/andres/hackmty26/audio")
    parser.add_argument("--salida", default=str(SALIDA_POR_DEFECTO))
    parser.add_argument(
        "--reiniciar", action="store_true", help="ignora --salida existente y empieza de cero"
    )
    args = parser.parse_args()

    filas_val = leer_manifest(Path(args.manifest), split="val")
    print(f"Llamadas en split=val: {len(filas_val)} (distribucion real, sin balancear ni mezclar con train)")

    registros = recolectar_datos(filas_val, Path(args.audio_dir), Path(args.salida), args.reiniciar)
    print(f"\nTotal de llamadas con datos utilizables: {len(registros)}/{len(filas_val)}")

    if len(registros) < 10:
        print("Muy pocos registros utilizables para validar de forma confiable. Saliendo.")
        return

    reportar_accuracy_eer(registros)
    reportar_brier(registros)
    reportar_reliability_diagram(registros)
    reportar_barrido_umbral(registros)


if __name__ == "__main__":
    main()
