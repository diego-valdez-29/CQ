"""
calibracion/acustico_ml_robusto.py

Reentrena el clasificador acústico con data augmentation (ruido + ganancia
variable) SOLO en train, para forzar al modelo a no depender del artefacto
de banda estrecha que ya identificamos (colapso a score=0.088 bajo ruido).

Ajusta EXTRAER_FEATURES() abajo para que coincida exactamente con la que
ya usa acustico_ml_entrenado.py -- si tu función real vive ahí, impórtala
en vez de duplicarla.
"""

import csv
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

MANIFEST = os.path.expanduser("~/hackmty26/manifest.csv")
AUDIO_DIR = os.path.expanduser("~/Downloads/audio")
N_AUMENTACIONES_POR_LLAMADA = 2  # + el original = 3 versiones por llamada de train


def cargar_canales(ruta_audio):
    data, sr = sf.read(ruta_audio, always_2d=True)
    if data.shape[1] == 1:
        return data[:, 0], sr
    return data[:, 0], sr  # solo el caller


def extraer_features(audio, sr, n_mfcc=20):
    """Ajusta esto si tu versión real difiere (debe dar el mismo shape
    que acustico_ml_entrenado.py para que los resultados sean comparables)."""
    mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = []
    for bloque in (mfcc, delta, delta2):
        feats.append(bloque.mean(axis=1))
        feats.append(bloque.var(axis=1))
    return np.concatenate(feats)


def aumentar_audio(audio, rng):
    """Genera una versión degradada: ruido blanco de SNR aleatorio + ganancia."""
    snr_db = rng.uniform(10, 30)
    rms_senal = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-9
    rms_ruido = rms_senal / (10 ** (snr_db / 20))
    ruido = rng.normal(0, rms_ruido, len(audio)).astype(np.float32)
    ganancia = rng.uniform(0.6, 1.4)
    return (audio.astype(np.float32) * ganancia) + ruido


def main():
    rng = np.random.default_rng(0)
    filas = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    train_rows = [r for r in filas if r["split"] == "train"]
    val_rows = [r for r in filas if r["split"] == "val"]

    print(f"train={len(train_rows)}  val={len(val_rows)}")
    print("Extrayendo features de TRAIN (con aumentaciones)...")

    X_train, y_train = [], []
    for i, row in enumerate(train_rows, 1):
        ruta = os.path.join(AUDIO_DIR, row["anon_id"] + ".wav")
        try:
            audio, sr = cargar_canales(ruta)
        except Exception as e:
            print(f"  [{i}/{len(train_rows)}] SKIP {row['anon_id']}: {e}")
            continue
        label = 1 if row["label"] == "synthetic" else 0

        # Original, sin degradar
        X_train.append(extraer_features(audio, sr))
        y_train.append(label)

        # Versiones aumentadas
        for _ in range(N_AUMENTACIONES_POR_LLAMADA):
            audio_aug = aumentar_audio(audio, rng)
            X_train.append(extraer_features(audio_aug, sr))
            y_train.append(label)

        if i % 50 == 0:
            print(f"  [{i}/{len(train_rows)}] procesadas")

    print(f"Total muestras de entrenamiento (con augmentation): {len(X_train)}")
    print("Extrayendo features de VAL (SIN degradar, para comparar limpio)...")

    X_val, y_val, ids_val = [], [], []
    for i, row in enumerate(val_rows, 1):
        ruta = os.path.join(AUDIO_DIR, row["anon_id"] + ".wav")
        try:
            audio, sr = cargar_canales(ruta)
        except Exception as e:
            print(f"  [{i}/{len(val_rows)}] SKIP {row['anon_id']}: {e}")
            continue
        X_val.append(extraer_features(audio, sr))
        y_val.append(1 if row["label"] == "synthetic" else 0)
        ids_val.append(row["anon_id"])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    print("\n=== Entrenando LR con augmentation ===")
    lr = LogisticRegression(max_iter=2000, random_state=0)
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_val_s)
    acc_lr = accuracy_score(y_val, pred_lr)
    cm_lr = confusion_matrix(y_val, pred_lr)
    print(f"Accuracy val (LR, con augmentation): {acc_lr:.3f}")
    print(f"Matriz [[TN,FP],[FN,TP]]: {cm_lr.tolist()}")

    print("\n=== Entrenando RF con augmentation ===")
    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0)
    rf.fit(X_train, y_train)  # RF no necesita scaler
    pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, pred_rf)
    cm_rf = confusion_matrix(y_val, pred_rf)
    print(f"Accuracy val (RF, con augmentation): {acc_rf:.3f}")
    print(f"Matriz [[TN,FP],[FN,TP]]: {cm_rf.tolist()}")

    print("\n=== Comparación contra el modelo SIN augmentation (ya descartado) ===")
    print("LR sin augmentation: accuracy=0.972 (colapsó bajo ruido)")
    print("RF sin augmentation: accuracy=0.958 (colapsó parcialmente bajo ruido)")
    print(f"LR con augmentation: accuracy={acc_lr:.3f}")
    print(f"RF con augmentation: accuracy={acc_rf:.3f}")

    # Guardar para la siguiente prueba de robustez
    import pickle
    with open("calibracion/modelo_acustico_robusto.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "lr": lr, "rf": rf}, f)
    print("\nModelos guardados en calibracion/modelo_acustico_robusto.pkl")
    print("SIGUIENTE PASO OBLIGATORIO: correr la prueba de robustez sobre estos")
    print("modelos antes de considerar integrarlos a fusion.py.")


if __name__ == "__main__":
    main()