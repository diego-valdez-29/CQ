"""
calibracion/probar_robustez_ml_robusto.py

Paso 2 del diagnostico de acustico_ml_robusto.py: el Paso 1
(diagnostico_features_augmentation.py) confirmo que aumentar_audio() SI
cambia las features de forma sustancial (diferencia media 156.5, 0/120
features sin cambio) -- asi que la accuracy identica (0.972 LR / 0.958 RF)
con y sin augmentation no se explica por dilucion de la augmentation en la
agregacion. Queda la otra hipotesis: el modelo sigue encontrando la misma
frontera de decision aunque vea datos de entrenamiento distintos.

Este script corre la prueba de robustez COMPLETA (las 71 llamadas de val,
no solo 10) sobre el modelo CON augmentation (calibracion/modelo_acustico_robusto.pkl),
con las mismas 3 degradaciones ya usadas en estres_condiciones_ml_acustico.py
(ruido blanco SNR~20dB, codec mp3 8kbps, downsample 8kHz-4kHz-8kHz, via
generar_version() de estres_condiciones.py -- no se reimplementan), y compara
contra el resultado ya documentado del modelo SIN augmentation (colapso a
score=0.088 constante bajo ruido en las 10 llamadas de prueba).
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acustico_ml_robusto import AUDIO_DIR, MANIFEST, cargar_canales, extraer_features
from estres_condiciones import AUDIO_SALIDA_DIR_POR_DEFECTO, NOMBRES_DEGRADACIONES, generar_version

UMBRAL_DECISION = 0.5

with open(Path(__file__).resolve().parent / "modelo_acustico_robusto.pkl", "rb") as f:
    m = pickle.load(f)
scaler, lr, rf = m["scaler"], m["lr"], m["rf"]

filas = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
val_rows = [r for r in filas if r["split"] == "val"]
print(f"Llamadas en split=val: {len(val_rows)}")

audio_dir = Path(AUDIO_DIR)
audio_salida_dir = AUDIO_SALIDA_DIR_POR_DEFECTO
audio_salida_dir.mkdir(parents=True, exist_ok=True)


def scores_desde_audio(audio: np.ndarray, sr: int) -> dict:
    feat = extraer_features(audio, sr).reshape(1, -1)
    score_lr = float(lr.predict_proba(scaler.transform(feat))[0, 1])
    score_rf = float(rf.predict_proba(feat)[0, 1])
    return {
        "lr": {"score": score_lr, "is_synthetic": score_lr >= UMBRAL_DECISION},
        "rf": {"score": score_rf, "is_synthetic": score_rf >= UMBRAL_DECISION},
    }


registros = []
y_true = []
pred_lr_orig, pred_rf_orig = [], []

for i, row in enumerate(val_rows, 1):
    anon_id = row["anon_id"]
    ruta_original = audio_dir / f"{anon_id}.wav"
    if not ruta_original.exists():
        print(f"  [{i}/{len(val_rows)}] {anon_id}: audio no encontrado, se omite")
        continue

    try:
        audio_orig, sr_orig = cargar_canales(str(ruta_original))
        resultados = {"original": scores_desde_audio(audio_orig, sr_orig)}

        for nombre_deg in NOMBRES_DEGRADACIONES:
            ruta_degradada = audio_salida_dir / f"{anon_id}_{nombre_deg}.wav"
            if not ruta_degradada.exists():
                generar_version(nombre_deg, ruta_original, ruta_degradada, anon_id)
            audio_deg, sr_deg = cargar_canales(str(ruta_degradada))
            resultados[nombre_deg] = scores_desde_audio(audio_deg, sr_deg)
    except Exception as exc:
        print(f"  [{i}/{len(val_rows)}] {anon_id}: FALLO ({exc!r}), se omite")
        continue

    label = row["label"]
    y = 1 if label == "synthetic" else 0
    y_true.append(y)
    pred_lr_orig.append(int(resultados["original"]["lr"]["is_synthetic"]))
    pred_rf_orig.append(int(resultados["original"]["rf"]["is_synthetic"]))

    registros.append({"anon_id": anon_id, "label": label, "resultados": resultados})
    if i % 10 == 0 or i == len(val_rows):
        print(f"  [{i}/{len(val_rows)}] procesadas")

n = len(registros)
y_true = np.array(y_true)
pred_lr_orig = np.array(pred_lr_orig)
pred_rf_orig = np.array(pred_rf_orig)

acc_lr_orig = float((y_true == pred_lr_orig).mean())
acc_rf_orig = float((y_true == pred_rf_orig).mean())

print(f"\n{'=' * 70}")
print(f"Sanity check: accuracy sobre val SIN degradar (n={n}), umbral={UMBRAL_DECISION}")
print(f"{'=' * 70}")
print(f"  LR: {acc_lr_orig:.3f}  (referencia reportada por acustico_ml_robusto.py: 0.972)")
print(f"  RF: {acc_rf_orig:.3f}  (referencia reportada por acustico_ml_robusto.py: 0.958)")

print(f"\n{'=' * 70}")
print(f"RESUMEN DE ROBUSTEZ -- modelo CON augmentation (n={n} llamadas de val, umbral={UMBRAL_DECISION})")
print(f"{'=' * 70}")
for nombre_modelo in ("lr", "rf"):
    print(f"\n  Modelo: {nombre_modelo}")
    for nombre_deg in NOMBRES_DEGRADACIONES:
        cambios = sum(
            1
            for r in registros
            if r["resultados"][nombre_deg][nombre_modelo]["is_synthetic"]
            != r["resultados"]["original"][nombre_modelo]["is_synthetic"]
        )
        print(f"    {nombre_deg:<18}: {cambios}/{n} llamadas cambiaron de clasificacion")

print(
    "\nComparacion contra el modelo SIN augmentation (calibracion/estres_condiciones_ml_acustico.py, n=10):"
)
print("  ruido             : LR 6/10, RF 4/10  (LR colapsaba a score=0.088 constante en las 10)")
print("  codec_agresivo    : LR 1/10, RF 2/10")
print("  downsample_extra  : LR 2/10, RF 2/10")
print("  (referencia: comportamiento en produccion = 4/10, 0/10, 0/10 respectivamente)")
