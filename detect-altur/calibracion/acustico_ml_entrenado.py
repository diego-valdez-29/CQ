"""acustico_ml_entrenado.py — entrena un clasificador de ML real (no
heuristica) para la senal acustica, usando MFCC + delta + delta-delta sobre
el canal caller de las llamadas reales del dataset.

Metodologia:

1. Extrae, por llamada, MFCC (N_MFCC coeficientes) + delta + delta-delta del
   canal caller preprocesado igual que DetectorAcustico (resample a 16kHz,
   normalizado a pico). Para no sobreajustar con solo ~282 llamadas de train,
   NO se usan features frame-by-frame: cada bloque (mfcc, delta, delta2) se
   agrega a media y varianza por coeficiente a lo largo del tiempo, dando un
   vector fijo de 6*N_MFCC valores por llamada.

2. Entrena regresion logistica (StandardScaler + LogisticRegression,
   regularizacion L2 por defecto de sklearn) sobre esas ~282 llamadas de
   split=train contra el label real.

3. Valida en las 71 llamadas de split=val (nunca vistas en el entrenamiento)
   -- accuracy, EER, matriz de confusion -- y compara contra:
     - comportamiento (produccion actual, scores crudos ya calculados en
       validacion_val_resultados.jsonl por validar_val.py)
     - DetectorAcustico, la heuristica MFCC original ya descartada
       (app/deteccion/acustico.py), evaluada aqui mismo sobre val con el
       mismo audio ya cargado (no estaba medida sobre val en ningun lado).

4. Si la regresion logistica resulta prometedora en val (accuracy > 0.65 y
   AUC > 0.5, es decir separa en la direccion correcta), entrena tambien un
   Random Forest pequeno (pocos arboles, profundidad limitada) sobre las
   mismas features, para ver si un modelo un poco mas expresivo mejora sin
   sobreajustar obviamente (train vs val).

No modifica fusion.py ni main.py -- esto es exploratorio, igual que
acustico_wav2vec.py. Si el resultado es bueno, la decision de meterlo al
pipeline de produccion se toma aparte.

Cachea features extraidas (MFCC agregados + score de la heuristica original)
en jsonl, incrementalmente, para poder reanudar sin volver a decodificar
audio si se interrumpe.

Uso:
    python calibracion/acustico_ml_entrenado.py
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import librosa.feature
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.acustico import DetectorAcustico
from app.main import cargar_canales

SR_OBJETIVO = DetectorAcustico.SR_OBJETIVO
N_MFCC = 20

MANIFEST_POR_DEFECTO = "/home/andres/hackmty26/manifest.csv"
AUDIO_DIR_POR_DEFECTO = "/home/andres/hackmty26/audio"
CACHE_POR_DEFECTO = Path(__file__).resolve().parent / "acustico_ml_features.jsonl"
COMPORTAMIENTO_JSONL = Path(__file__).resolve().parent / "validacion_val_resultados.jsonl"
UMBRAL_COMPORTAMIENTO_PRODUCCION = 0.58

UMBRAL_DECISION = 0.5  # umbral de decision para LR/RF/heuristica: no se recalibra aqui,
# solo se compara a igualdad de condiciones (score >= 0.5 -> sintetico).

ACCURACY_MINIMA_PROMETEDORA = 0.65


def leer_manifest(ruta_manifest: Path) -> list[dict]:
    with open(ruta_manifest, newline="") as f:
        return list(csv.DictReader(f))


def cargar_cache(ruta_cache: Path) -> dict[str, dict]:
    if not ruta_cache.exists():
        return {}
    registros = {}
    for linea in ruta_cache.read_text().splitlines():
        linea = linea.strip()
        if linea:
            r = json.loads(linea)
            registros[r["anon_id"]] = r
    return registros


def extraer_features_mfcc(audio_preprocesado: np.ndarray) -> np.ndarray:
    """MFCC + delta + delta-delta, agregados (media y varianza por
    coeficiente en el tiempo) para no sobreajustar con features
    frame-by-frame contra tan pocas llamadas.
    """
    mfcc = librosa.feature.mfcc(y=audio_preprocesado, sr=SR_OBJETIVO, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    partes = []
    for bloque in (mfcc, delta, delta2):
        partes.append(np.mean(bloque, axis=1))
        partes.append(np.var(bloque, axis=1))
    return np.concatenate(partes)


def recolectar_features(
    filas: list[dict], audio_dir: Path, ruta_cache: Path, reiniciar: bool
) -> dict[str, dict]:
    cache = {} if reiniciar else cargar_cache(ruta_cache)
    if cache:
        print(f"Reanudando: {len(cache)} llamadas ya en cache ({ruta_cache}), se saltan.")

    pendientes = [f for f in filas if f["anon_id"] not in cache]
    if not pendientes:
        return cache

    detector_heuristica = DetectorAcustico()
    modo_escritura = "a" if (cache and not reiniciar) else "w"
    archivo_salida = open(ruta_cache, modo_escritura)

    for i, fila in enumerate(pendientes, start=1):
        anon_id = fila["anon_id"]
        ruta_audio = audio_dir / f"{anon_id}.wav"
        if not ruta_audio.exists():
            print(f"  [{i}/{len(pendientes)}] {anon_id}: audio no encontrado, se omite")
            continue

        try:
            canal_caller, _, sr = cargar_canales(str(ruta_audio))
            audio, duracion_s = DetectorAcustico._preprocesar(canal_caller, sr)
            if not DetectorAcustico._verificar_condiciones_analisis(audio):
                print(f"  [{i}/{len(pendientes)}] {anon_id}: audio insuficiente, se omite")
                continue

            features = extraer_features_mfcc(audio)

            features_heur = DetectorAcustico._extraer_caracteristicas(audio)
            score_heuristico = detector_heuristica._score_heuristico(features_heur)
        except Exception as exc:
            print(f"  [{i}/{len(pendientes)}] {anon_id}: FALLO ({exc!r}), se omite")
            continue

        registro = {
            "anon_id": anon_id,
            "label": fila["label"],
            "split": fila["split"],
            "duracion_s": duracion_s,
            "features_mfcc": features.tolist(),
            "score_heuristico_original": score_heuristico,
        }
        cache[anon_id] = registro
        archivo_salida.write(json.dumps(registro) + "\n")
        archivo_salida.flush()
        print(f"  [{i}/{len(pendientes)}] {anon_id} ({fila['label']}, {fila['split']}): ok")

    archivo_salida.close()
    return cache


def _matriz_y_eer(y: np.ndarray, score: np.ndarray, umbral: float) -> dict:
    from sklearn.metrics import confusion_matrix, roc_curve

    pred = (score >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    accuracy = (tn + tp) / len(y)

    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)

    return {
        "accuracy": accuracy,
        "eer": eer,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def imprimir_resultado(nombre: str, resultado: dict) -> None:
    print(f"\n--- {nombre} ---")
    print(f"  Accuracy: {resultado['accuracy']:.3f}")
    print(f"  EER:      {resultado['eer']:.3f}")
    print(
        f"  Matriz de confusion [[TN,FP],[FN,TP]] = "
        f"[[{resultado['tn']}, {resultado['fp']}], [{resultado['fn']}, {resultado['tp']}]]"
    )


def cargar_comportamiento_val() -> dict:
    """Reusa los scores crudos de comportamiento ya calculados sobre val por
    validar_val.py -- no vuelve a correr el detector de comportamiento.
    """
    y, score = [], []
    for linea in COMPORTAMIENTO_JSONL.read_text().splitlines():
        linea = linea.strip()
        if not linea:
            continue
        r = json.loads(linea)
        y.append(1 if r["label"] == "synthetic" else 0)
        score.append(r["score_comportamiento"])
    return _matriz_y_eer(
        np.array(y), np.array(score), UMBRAL_COMPORTAMIENTO_PRODUCCION
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default=MANIFEST_POR_DEFECTO)
    parser.add_argument("--audio-dir", default=AUDIO_DIR_POR_DEFECTO)
    parser.add_argument("--cache", default=str(CACHE_POR_DEFECTO))
    parser.add_argument("--reiniciar", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    filas = leer_manifest(Path(args.manifest))
    print(f"Manifest: {len(filas)} llamadas totales (train+val)")

    cache = recolectar_features(filas, Path(args.audio_dir), Path(args.cache), args.reiniciar)

    registros_train = [r for r in cache.values() if r["split"] == "train"]
    registros_val = [r for r in cache.values() if r["split"] == "val"]
    print(f"\nUtilizables: train={len(registros_train)}  val={len(registros_val)}")

    if len(registros_train) < 20 or len(registros_val) < 10:
        print("Muy pocos registros utilizables. Saliendo.")
        return

    X_train = np.array([r["features_mfcc"] for r in registros_train])
    y_train = np.array([1 if r["label"] == "synthetic" else 0 for r in registros_train])
    X_val = np.array([r["features_mfcc"] for r in registros_val])
    y_val = np.array([1 if r["label"] == "synthetic" else 0 for r in registros_val])

    print(
        f"Balance train: {int(y_train.sum())} synthetic / {int((1 - y_train).sum())} human  "
        f"(n={len(y_train)})"
    )
    print(
        f"Balance val:   {int(y_val.sum())} synthetic / {int((1 - y_val).sum())} human  "
        f"(n={len(y_val)})"
    )

    # --- Paso 2/3: regresion logistica ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    modelo_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    modelo_lr.fit(X_train, y_train)

    proba_val_lr = modelo_lr.predict_proba(X_val)[:, 1]
    resultado_lr = _matriz_y_eer(y_val, proba_val_lr, UMBRAL_DECISION)
    auc_lr = float(roc_auc_score(y_val, proba_val_lr))

    proba_train_lr = modelo_lr.predict_proba(X_train)[:, 1]
    resultado_lr_train = _matriz_y_eer(y_train, proba_train_lr, UMBRAL_DECISION)

    # --- Heuristica original (DetectorAcustico), evaluada sobre val ---
    score_heur_val = np.array([r["score_heuristico_original"] for r in registros_val])
    resultado_heuristica = _matriz_y_eer(y_val, score_heur_val, UMBRAL_DECISION)

    # --- Comportamiento (produccion), reusando validar_val.py ---
    resultado_comportamiento = cargar_comportamiento_val()

    print("\n" + "=" * 70)
    print("RESULTADOS SOBRE split=val (n={})".format(len(y_val)))
    print("=" * 70)

    imprimir_resultado(
        f"Comportamiento (produccion, umbral={UMBRAL_COMPORTAMIENTO_PRODUCCION})",
        resultado_comportamiento,
    )
    imprimir_resultado(
        f"Heuristica acustica original (DetectorAcustico, umbral={UMBRAL_DECISION})",
        resultado_heuristica,
    )
    imprimir_resultado(
        f"Regresion logistica MFCC+delta+delta2 (umbral={UMBRAL_DECISION}, AUC={auc_lr:.3f})",
        resultado_lr,
    )
    print(
        f"\n  [chequeo de sobreajuste LR] accuracy train={resultado_lr_train['accuracy']:.3f} "
        f"vs val={resultado_lr['accuracy']:.3f} "
        f"(diff={resultado_lr_train['accuracy'] - resultado_lr['accuracy']:+.3f})"
    )

    # --- Paso 4: Random Forest, solo si LR resulto prometedora ---
    prometedora = resultado_lr["accuracy"] > ACCURACY_MINIMA_PROMETEDORA and auc_lr > 0.5
    if not prometedora:
        print(
            f"\nLR no cruzo el umbral de 'prometedor' (accuracy > {ACCURACY_MINIMA_PROMETEDORA} "
            f"y AUC > 0.5; obtuvo accuracy={resultado_lr['accuracy']:.3f}, AUC={auc_lr:.3f}) "
            "-- no se entrena Random Forest."
        )
    else:
        from sklearn.ensemble import RandomForestClassifier

        modelo_rf = RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=0, class_weight="balanced"
        )
        modelo_rf.fit(X_train, y_train)

        proba_val_rf = modelo_rf.predict_proba(X_val)[:, 1]
        resultado_rf = _matriz_y_eer(y_val, proba_val_rf, UMBRAL_DECISION)
        auc_rf = float(roc_auc_score(y_val, proba_val_rf))

        proba_train_rf = modelo_rf.predict_proba(X_train)[:, 1]
        resultado_rf_train = _matriz_y_eer(y_train, proba_train_rf, UMBRAL_DECISION)

        imprimir_resultado(
            f"Random Forest (50 arboles, max_depth=4, umbral={UMBRAL_DECISION}, AUC={auc_rf:.3f})",
            resultado_rf,
        )
        print(
            f"\n  [chequeo de sobreajuste RF] accuracy train={resultado_rf_train['accuracy']:.3f} "
            f"vs val={resultado_rf['accuracy']:.3f} "
            f"(diff={resultado_rf_train['accuracy'] - resultado_rf['accuracy']:+.3f})"
        )

    print("\n" + "=" * 70)
    print("RESUMEN LADO A LADO (accuracy / EER sobre val, n={})".format(len(y_val)))
    print("=" * 70)
    print(f"  {'senal':<45} {'accuracy':>9} {'EER':>7}")
    print(f"  {'comportamiento (produccion)':<45} {resultado_comportamiento['accuracy']:>9.3f} {resultado_comportamiento['eer']:>7.3f}")
    print(f"  {'heuristica acustica original':<45} {resultado_heuristica['accuracy']:>9.3f} {resultado_heuristica['eer']:>7.3f}")
    print(f"  {'LR MFCC+delta+delta2 (nuevo)':<45} {resultado_lr['accuracy']:>9.3f} {resultado_lr['eer']:>7.3f}")
    if prometedora:
        print(f"  {'Random Forest MFCC+delta+delta2 (nuevo)':<45} {resultado_rf['accuracy']:>9.3f} {resultado_rf['eer']:>7.3f}")
    print(
        "\nRecordatorio: esto NO toca fusion.py ni main.py -- es exploratorio, igual que "
        "acustico_wav2vec.py. Umbral de decision usado aqui (0.5) es el default de LR/RF/heuristica, "
        "no un umbral recalibrado como el 0.58 de comportamiento en produccion."
    )


if __name__ == "__main__":
    main()
