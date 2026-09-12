"""calibrar_confidence.py — prueba si Platt scaling mejora la calibracion de
PROBABILIDAD (que tan bien `confidence` refleja P(sintetico), medido con
Brier score) del score crudo de `comportamiento` sobre las 71 llamadas de
`val`. NO es calibracion de decision: no toca pesos, umbral, ni
`fusion.py`/`main.py` — validar_val.py ya establecio que a umbral=0.56 el
Brier score en val es 0.236 (cerca de 0.25 = tan bueno como predecir
siempre 0.5) y que el reliability diagram esta lejos de la diagonal ideal.
Este script prueba UNA correccion simple para eso: Platt scaling.

Usa `sklearn.calibration._SigmoidCalibration` (la implementacion estandar de
Platt, 1999: ajusta a y b tal que `calibrada = 1 / (1 + exp(a*score + b))`,
por regresion logistica 1D sobre el score crudo) -- es literalmente lo que
usa `CalibratedClassifierCV(method="sigmoid")` por dentro, aplicado aqui
directamente sobre nuestro score ya calculado en vez de envolver un
clasificador completo.

Reutiliza `leer_manifest` y `recolectar_datos` de validar_val.py: si
`calibracion/validacion_val_resultados.jsonl` ya tiene los 71 scores
crudos (de una corrida anterior de validar_val.py), los usa tal cual: no
vuelve a correr VAD. Si faltan llamadas, las calcula y las agrega al mismo
cache incremental.

Reporta Brier score antes/despues y, si la mejora es clara, imprime un
snippet de codigo (los 2 parametros nuevos, a y b) listo para aplicar
dentro de `Fusion.combinar()` o justo antes de retornar `confidence` en
`app/main.py`. No aplica el cambio: solo lo muestra.

Uso tipico:
    python calibrar_confidence.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import _SigmoidCalibration

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validar_val import BUCKETS, SALIDA_POR_DEFECTO, leer_manifest, recolectar_datos

MEJORA_RELATIVA_CLARA = 0.05  # 5% de reduccion en Brier score se considera "mejora clara"


def brier_score(confianza: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((confianza - y) ** 2))


def ajustar_platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    modelo = _SigmoidCalibration()
    modelo.fit(scores, y)
    calibrado = modelo.predict(scores)
    return float(modelo.a_), float(modelo.b_), calibrado


def reportar_reliability_diagram(titulo: str, confs: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{titulo}")
    for lo, hi in BUCKETS:
        en_bucket = (confs >= lo) & (confs <= hi) if hi == 1.0 else (confs >= lo) & (confs < hi)
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


def imprimir_snippet(a: float, b: float, brier_antes: float, brier_despues: float) -> None:
    print(
        "\n--- snippet sugerido (NO aplicado por este script) ---\n"
        "Opcion A: dentro de Fusion.combinar(), calibrando la confidence ya fusionada\n"
        "(app/deteccion/fusion.py):\n"
        "\n"
        "    import numpy as np\n"
        "\n"
        "    class Fusion:\n"
        "        # Platt scaling ajustado sobre split val (n=71, calibracion/calibrar_confidence.py)\n"
        f"        PLATT_A = {a!r}\n"
        f"        PLATT_B = {b!r}\n"
        "\n"
        "        def combinar(self, senales):\n"
        "            ...\n"
        "            confidence_calibrada = 1.0 / (1.0 + np.exp(self.PLATT_A * confidence + self.PLATT_B))\n"
        "            return ResultadoDeteccion(\n"
        "                is_synthetic=confidence_calibrada >= self.umbral,\n"
        "                confidence=confidence_calibrada,\n"
        "                senales=senales,\n"
        "            )\n"
        "\n"
        "Opcion B: justo antes de retornar confidence en app/main.py (_respuesta()), sin tocar\n"
        "fusion.py -- mas aislado si se quiere poder revertir sin tocar la clase compartida:\n"
        "\n"
        "    import numpy as np\n"
        "\n"
        "    # Platt scaling ajustado sobre split val (n=71, calibracion/calibrar_confidence.py)\n"
        f"    PLATT_A = {a!r}\n"
        f"    PLATT_B = {b!r}\n"
        "\n"
        "    def _calibrar_confidence(confidence: float) -> float:\n"
        "        return 1.0 / (1.0 + np.exp(PLATT_A * confidence + PLATT_B))\n"
        "\n"
        "IMPORTANTE: si se aplica, el umbral de decision (hoy 0.58) tambien queda en la escala\n"
        "calibrada, no en la escala del score crudo -- habria que re-barrer el umbral sobre\n"
        "confidence_calibrada, no reusar 0.58 tal cual.\n"
        "\n"
        f"Brier score: {brier_antes:.3f} (antes) -> {brier_despues:.3f} (despues)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default="/home/andres/hackmty26/manifest.csv")
    parser.add_argument("--audio-dir", default="/home/andres/hackmty26/audio")
    parser.add_argument("--scores-jsonl", default=str(SALIDA_POR_DEFECTO))
    args = parser.parse_args()

    filas_val = leer_manifest(Path(args.manifest), split="val")
    print(f"Llamadas en split=val: {len(filas_val)}")

    registros = recolectar_datos(filas_val, Path(args.audio_dir), Path(args.scores_jsonl), reiniciar=False)
    print(f"Total de llamadas con scores utilizables: {len(registros)}/{len(filas_val)}")
    if len(registros) < 10:
        print("Muy pocos registros utilizables para calibrar Platt scaling de forma confiable. Saliendo.")
        return

    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])
    scores = np.array([r["score_comportamiento"] for r in registros])

    brier_antes = brier_score(scores, y)
    print(f"\n=== Paso 1: Brier score ANTES de Platt scaling (score crudo de comportamiento, n={len(registros)}) ===")
    print(f"Brier score: {brier_antes:.3f}")

    a, b, calibrado = ajustar_platt(scores, y)
    print("\n=== Paso 2: ajuste de Platt scaling (sklearn.calibration._SigmoidCalibration) ===")
    print(f"a={a:.5f}  b={b:.5f}")
    print("confidence_calibrada = 1 / (1 + exp(a * score + b))")

    brier_despues = brier_score(calibrado, y)
    print(f"\n=== Paso 3: Brier score DESPUES de Platt scaling ===")
    print(f"Brier score: {brier_despues:.3f}")
    mejora_absoluta = brier_antes - brier_despues
    mejora_relativa = mejora_absoluta / brier_antes if brier_antes > 0 else 0.0
    print(f"Mejora: {mejora_absoluta:+.3f} absoluto ({mejora_relativa:+.1%} relativo)")
    print(
        "AVISO: a y b se ajustaron y se evaluaron sobre las MISMAS 71 llamadas de val (no hay "
        "un split adicional para validar el ajuste de Platt de forma independiente) -- esta "
        "mejora es in-sample y probablemente algo optimista, igual que paso con el umbral "
        "recalibrado en val (ver README.md)."
    )

    reportar_reliability_diagram(
        "=== Paso 4: reliability diagram ANTES de Platt scaling ===", scores, y
    )
    reportar_reliability_diagram(
        "=== Paso 4: reliability diagram DESPUES de Platt scaling ===", calibrado, y
    )

    print(f"\n=== Paso 5: decision ===")
    if mejora_relativa >= MEJORA_RELATIVA_CLARA:
        print(
            f"Mejora clara (>= {MEJORA_RELATIVA_CLARA:.0%} relativo): vale la pena considerar "
            "aplicar Platt scaling."
        )
        imprimir_snippet(a, b, brier_antes, brier_despues)
    else:
        print(
            f"Mejora NO clara (< {MEJORA_RELATIVA_CLARA:.0%} relativo): no se recomienda aplicar "
            "Platt scaling en produccion todavia -- el riesgo de sobreajustar 2 parametros mas "
            "sobre las mismas 71 llamadas de val no se justifica con esta ganancia."
        )


if __name__ == "__main__":
    main()
