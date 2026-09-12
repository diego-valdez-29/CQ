"""estres_condiciones_ml_acustico.py — prueba de robustez (NO calibracion)
del modelo de ML entrenado en acustico_ml_entrenado.py (LR y Random Forest
sobre MFCC+delta+delta2) ante las mismas 3 condiciones de audio degradado ya
usadas en estres_condiciones.py para DetectorComportamiento.

Motivacion: acustico_ml_entrenado.py obtuvo accuracy=0.972 (LR) y 0.958 (RF)
sobre val, muy por encima de comportamiento (0.789) y de la heuristica MFCC
original (0.141) -- una separacion casi perfecta que contradice lo que el
propio README reporta para MFCC crudo en train (no separaba nada). La
verificacion de fuga por StandardScaler ya descarto que sea un problema de
fit/transform train-val. Queda una hipotesis distinta y mas grave: que el
modelo aprendio una huella de COMO SE GENERO/GUARDO el dataset (shortcut
learning) en vez de una senal real de sintesis de voz -- exactamente el
patron ya descartado para wav2vec2 (ver calibracion/acustico_wav2vec.py).

Esta es la prueba: si mp3 a 8kbps (una degradacion minima, cualquier llamada
telefonica real ya la sufre) tira la accuracy del modelo, la senal no es
robusta y no vale para produccion, sin importar que tan bien le fue en val
sin degradar.

Reutiliza EXACTAMENTE:
  - La misma muestra de 10 llamadas de val (5 human / 5 synthetic,
    semilla=0) via leer_manifest/muestrear_balanceado de calibrar.py --
    igual que estres_condiciones.py, para poder comparar lado a lado.
  - Las mismas 3 degradaciones (ruido blanco SNR~20dB, codec mp3 8kbps,
    downsample 8kHz->4kHz->8kHz) via generar_version() de
    estres_condiciones.py -- no se reimplementan.
  - La misma extraccion de features (extraer_features_mfcc, N_MFCC=20) y el
    mismo preprocesamiento (DetectorAcustico._preprocesar) de
    acustico_ml_entrenado.py.

Entrena LR y RF desde cero sobre las 282 llamadas de train cacheadas en
acustico_ml_features.jsonl (mismos hiperparametros que
acustico_ml_entrenado.py) -- no se serializo el modelo en ese script, asi
que se re-entrena aqui con los mismos datos y semillas.

No modifica fusion.py, main.py ni acustico_ml_entrenado.py.

Uso:
    python estres_condiciones_ml_acustico.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.deteccion.acustico import DetectorAcustico
from app.main import cargar_canales
from acustico_ml_entrenado import CACHE_POR_DEFECTO, N_MFCC, UMBRAL_DECISION, extraer_features_mfcc
from calibrar import leer_manifest, muestrear_balanceado
from estres_condiciones import (
    AUDIO_SALIDA_DIR_POR_DEFECTO,
    MUESTRA,
    NOMBRES_DEGRADACIONES,
    SEMILLA,
    generar_version,
)

MODELOS = ("lr", "rf")


def entrenar_modelos(ruta_cache: Path) -> dict:
    """Re-entrena LR y RF sobre las 282 llamadas de train ya cacheadas por
    acustico_ml_entrenado.py -- mismos hiperparametros, mismas semillas.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    registros = [json.loads(l) for l in ruta_cache.read_text().splitlines() if l.strip()]
    train = [r for r in registros if r["split"] == "train"]
    if len(train) < 20:
        raise RuntimeError(
            f"Muy pocas llamadas de train en {ruta_cache} ({len(train)}). "
            "Corre primero acustico_ml_entrenado.py para generar el cache de features."
        )

    X_train = np.array([r["features_mfcc"] for r in train])
    y_train = np.array([1 if r["label"] == "synthetic" else 0 for r in train])

    modelo_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    modelo_lr.fit(X_train, y_train)

    modelo_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0, class_weight="balanced")
    modelo_rf.fit(X_train, y_train)

    print(f"Modelos re-entrenados sobre {len(train)} llamadas de train (cache: {ruta_cache})")
    return {"lr": modelo_lr, "rf": modelo_rf}


def extraer_features_de_audio(ruta: Path) -> np.ndarray:
    canal_caller, _, sr = cargar_canales(str(ruta))
    audio, _ = DetectorAcustico._preprocesar(canal_caller, sr)
    return extraer_features_mfcc(audio)


def evaluar_version(modelos: dict, ruta: Path) -> dict:
    features = extraer_features_de_audio(ruta).reshape(1, -1)
    resultado = {}
    for nombre, modelo in modelos.items():
        score = float(modelo.predict_proba(features)[0, 1])
        resultado[nombre] = {"score": score, "is_synthetic": score >= UMBRAL_DECISION}
    return resultado


def procesar_llamada(modelos: dict, fila: dict, audio_dir: Path, audio_salida_dir: Path) -> dict | None:
    anon_id = fila["anon_id"]
    ruta_original = audio_dir / f"{anon_id}.wav"
    if not ruta_original.exists():
        print(f"  {anon_id}: audio no encontrado en {ruta_original}, se omite")
        return None

    resultados = {}
    try:
        resultados["original"] = evaluar_version(modelos, ruta_original)
        for nombre_deg in NOMBRES_DEGRADACIONES:
            ruta_degradada = audio_salida_dir / f"{anon_id}_{nombre_deg}.wav"
            if not ruta_degradada.exists():
                generar_version(nombre_deg, ruta_original, ruta_degradada, anon_id)
            resultados[nombre_deg] = evaluar_version(modelos, ruta_degradada)
    except Exception as exc:
        print(f"  {anon_id}: FALLO ({exc!r}), se omite")
        return None

    return {"anon_id": anon_id, "label": fila["label"], "resultados": resultados}


def reportar_llamada(registro: dict) -> None:
    anon_id, label, resultados = registro["anon_id"], registro["label"], registro["resultados"]
    original = resultados["original"]
    print(f"\n{anon_id} (label={label}):")
    for nombre_modelo in MODELOS:
        o = original[nombre_modelo]
        print(f"  [{nombre_modelo}] original          : score={o['score']:.3f} is_synthetic={o['is_synthetic']}")
        for nombre_deg in NOMBRES_DEGRADACIONES:
            r = resultados[nombre_deg][nombre_modelo]
            cambio = "CAMBIO" if r["is_synthetic"] != o["is_synthetic"] else "igual"
            print(
                f"  [{nombre_modelo}] {nombre_deg:<18}: score={r['score']:.3f} "
                f"is_synthetic={r['is_synthetic']} ({cambio} vs original)"
            )


def reportar_resumen(registros: list[dict]) -> None:
    n = len(registros)
    print(f"\n=== Resumen de robustez (n={n} llamadas, umbral={UMBRAL_DECISION}) ===")
    for nombre_modelo in MODELOS:
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
        "\nComparacion de referencia (calibracion/estres_condiciones.py, DetectorComportamiento): "
        "codec_agresivo=0/10, downsample_extra=0/10, ruido=4/10 (ver README.md)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default="/home/andres/hackmty26/manifest.csv")
    parser.add_argument("--audio-dir", default="/home/andres/hackmty26/audio")
    parser.add_argument("--audio-salida-dir", default=str(AUDIO_SALIDA_DIR_POR_DEFECTO))
    parser.add_argument("--cache-features", default=str(CACHE_POR_DEFECTO))
    parser.add_argument("--muestra", type=int, default=MUESTRA)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args()

    audio_salida_dir = Path(args.audio_salida_dir)
    audio_salida_dir.mkdir(parents=True, exist_ok=True)

    modelos = entrenar_modelos(Path(args.cache_features))

    filas_val = leer_manifest(Path(args.manifest), split="val")
    muestra = muestrear_balanceado(filas_val, args.muestra, args.semilla)
    print(
        f"Muestra balanceada: {len(muestra)} llamadas "
        f"({args.muestra // 2} human / {args.muestra // 2} synthetic), semilla={args.semilla} "
        "(misma seleccion que estres_condiciones.py)"
    )
    print(f"WAVs degradados se guardan/reusan en: {audio_salida_dir}")

    registros = []
    for fila in muestra:
        registro = procesar_llamada(modelos, fila, Path(args.audio_dir), audio_salida_dir)
        if registro is not None:
            registros.append(registro)
            reportar_llamada(registro)

    print(f"\nTotal de llamadas procesadas: {len(registros)}/{len(muestra)}")
    if not registros:
        print("Sin registros utilizables. Saliendo.")
        return

    reportar_resumen(registros)


if __name__ == "__main__":
    main()
