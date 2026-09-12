"""modelo_costo.py — traduce la matriz de confusion sobre split=val a un costo
de negocio simplificado para Altur (cobro por exito de cobranza), en vez de
optimizar solo accuracy.

Usa los scores crudos ya calculados en validacion_val_resultados.jsonl (n=71,
ver validar_val.py) — no vuelve a correr el pipeline de audio. Como el peso de
produccion es {comportamiento: 1.0, resto: 0.0} (ver README.md), la confidence
fusionada es identica al score_comportamiento crudo (Fusion.combinar hace un
promedio ponderado de una sola senal con peso != 0): por eso aqui se usa el
score directamente en vez de instanciar Fusion.

Supuestos de costo (explicitos, ajustables abajo):

- Falso positivo (humano marcado como sintetico): la llamada de cobranza
  legitima se rechaza o se escala innecesariamente -> se pierde (o se retrasa)
  ese cobro. Costo relativo = 1.
- Falso negativo (sintetico marcado como humano): un fraude pasa sin
  detectar -> Altur trata como legitima una gestion de cobranza que no lo es.
  Costo relativo = COSTO_FN_POR_FP veces el de un FP (por defecto 5x: se
  asume que dejar pasar un fraude sale mas caro que escalar de mas una
  llamada real, pero es un supuesto de negocio, no un dato medido).

Barre el umbral de UMBRAL_SWEEP_INICIO a UMBRAL_SWEEP_FIN y reporta, para cada
uno, la matriz de confusion, accuracy y costo total — y cual umbral minimiza
costo, comparado contra el 0.58 ya en produccion.

Uso:
    python modelo_costo.py
    python modelo_costo.py --costo-fn-por-fp 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

ENTRADA_POR_DEFECTO = Path(__file__).resolve().parent / "validacion_val_resultados.jsonl"

UMBRAL_PRODUCCION = 0.58
UMBRAL_SWEEP_INICIO = 0.30
UMBRAL_SWEEP_FIN = 0.70
UMBRAL_SWEEP_PASO = 0.02

# Costo relativo de un FN expresado en unidades de "un FP" (ver docstring).
COSTO_FN_POR_FP_DEFECTO = 5.0
COSTO_FP = 1.0


def cargar_datos(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    y, conf = [], []
    for linea in ruta.read_text().splitlines():
        linea = linea.strip()
        if not linea:
            continue
        r = json.loads(linea)
        y.append(1 if r["label"] == "synthetic" else 0)
        conf.append(r["score_comportamiento"])
    return np.array(y), np.array(conf)


def costo_en_umbral(y: np.ndarray, conf: np.ndarray, umbral: float, costo_fn: float) -> dict:
    pred = (conf >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    costo_total = fp * COSTO_FP + fn * costo_fn
    accuracy = (tn + tp) / len(y)
    return {
        "umbral": umbral,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy,
        "costo_total": costo_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--entrada", default=str(ENTRADA_POR_DEFECTO))
    parser.add_argument(
        "--costo-fn-por-fp",
        type=float,
        default=COSTO_FN_POR_FP_DEFECTO,
        help=f"costo relativo de un FN vs un FP (default {COSTO_FN_POR_FP_DEFECTO}:1)",
    )
    args = parser.parse_args()

    y, conf = cargar_datos(Path(args.entrada))
    n = len(y)
    print(f"Datos: n={n} llamadas de split=val (score_comportamiento crudo == confidence fusionada)")
    print(f"Supuesto de costo: FN = {args.costo_fn_por_fp:.1f} x FP  (FP={COSTO_FP:.1f}, FN={args.costo_fn_por_fp:.1f})")

    umbrales = np.round(np.arange(UMBRAL_SWEEP_INICIO, UMBRAL_SWEEP_FIN + 1e-9, UMBRAL_SWEEP_PASO), 2)
    filas = [costo_en_umbral(y, conf, u, args.costo_fn_por_fp) for u in umbrales]

    print(
        f"\n=== Barrido de umbral ({UMBRAL_SWEEP_INICIO:.2f}-{UMBRAL_SWEEP_FIN:.2f}, "
        f"paso {UMBRAL_SWEEP_PASO:.2f}) sobre split=val ==="
    )
    print(f"{'umbral':>7} {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4} {'accuracy':>9} {'costo':>8}")
    for f in filas:
        marca = "  <- produccion actual" if abs(f["umbral"] - UMBRAL_PRODUCCION) < 1e-9 else ""
        print(
            f"{f['umbral']:>7.2f} {f['tn']:>4} {f['fp']:>4} {f['fn']:>4} {f['tp']:>4} "
            f"{f['accuracy']:>9.3f} {f['costo_total']:>8.1f}{marca}"
        )

    actual = next(f for f in filas if abs(f["umbral"] - UMBRAL_PRODUCCION) < 1e-9)
    mejor_costo = min(f["costo_total"] for f in filas)
    candidatos_costo = [f for f in filas if abs(f["costo_total"] - mejor_costo) < 1e-9]
    optimo_costo = min(candidatos_costo, key=lambda f: abs(f["umbral"] - UMBRAL_PRODUCCION))

    mejor_acc = max(f["accuracy"] for f in filas)
    candidatos_acc = [f for f in filas if abs(f["accuracy"] - mejor_acc) < 1e-9]
    optimo_acc = min(candidatos_acc, key=lambda f: abs(f["umbral"] - UMBRAL_PRODUCCION))

    print(f"\n=== Comparacion ===")
    print(
        f"Umbral actual en produccion ({UMBRAL_PRODUCCION:.2f}): "
        f"accuracy={actual['accuracy']:.3f}  costo={actual['costo_total']:.1f}  "
        f"(FP={actual['fp']}, FN={actual['fn']})"
    )
    print(
        f"Umbral optimo por ACCURACY ({optimo_acc['umbral']:.2f}): "
        f"accuracy={optimo_acc['accuracy']:.3f}  costo={optimo_acc['costo_total']:.1f}  "
        f"(FP={optimo_acc['fp']}, FN={optimo_acc['fn']})"
    )
    print(
        f"Umbral optimo por COSTO   ({optimo_costo['umbral']:.2f}): "
        f"accuracy={optimo_costo['accuracy']:.3f}  costo={optimo_costo['costo_total']:.1f}  "
        f"(FP={optimo_costo['fp']}, FN={optimo_costo['fn']})"
    )

    ahorro = actual["costo_total"] - optimo_costo["costo_total"]
    if abs(optimo_costo["umbral"] - UMBRAL_PRODUCCION) < 1e-9:
        print(
            f"\nEl umbral actual ({UMBRAL_PRODUCCION:.2f}) YA minimiza el costo bajo este supuesto "
            f"({args.costo_fn_por_fp:.0f}:1) — no hay motivo para moverlo por costo."
        )
    else:
        direccion = "bajar" if optimo_costo["umbral"] < UMBRAL_PRODUCCION else "subir"
        print(
            f"\nBajo el supuesto {args.costo_fn_por_fp:.0f}:1, {direccion} el umbral de "
            f"{UMBRAL_PRODUCCION:.2f} a {optimo_costo['umbral']:.2f} ahorra {ahorro:.1f} unidades de costo "
            f"sobre estas {n} llamadas ({ahorro / n:.2f} por llamada) a cambio de "
            f"{'menos' if optimo_costo['fn'] < actual['fn'] else 'mas'} falsos negativos "
            f"({actual['fn']} -> {optimo_costo['fn']}) y "
            f"{'mas' if optimo_costo['fp'] > actual['fp'] else 'menos'} falsos positivos "
            f"({actual['fp']} -> {optimo_costo['fp']})."
        )
        print(
            "Nota: este umbral optimo se calculo sobre las mismas 71 llamadas de val que ya sirven de "
            "'conjunto de prueba' en el README — es un supuesto de costo explorable, no una recalibracion "
            "de produccion."
        )


if __name__ == "__main__":
    main()
