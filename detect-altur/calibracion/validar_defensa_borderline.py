"""validar_defensa_borderline.py — mide (NO implementa) si std_recuperacion
podria servir como desempate para las llamadas de val cuyo score de
comportamiento cae cerca del umbral de produccion.

Paso 1 reutiliza los scores YA calculados por validar_val.py
(validacion_val_resultados.jsonl, 71 llamadas de val, columna
score_comportamiento) -- no vuelve a correr el detector sobre las 71. Pero
esos resultados NO incluyen std_recuperacion (validar_val.py solo guarda
score_comportamiento; std_recuperacion se calcula dentro de
DetectorComportamiento.analizar() pero nunca se persiste a disco en ningun
script existente). Por eso, para el subconjunto pequeño de llamadas
"borderline" (Paso 2), este script SI vuelve a correr
DetectorComportamiento.analizar() sobre su audio, solo para leer
detalle["std_recuperacion"] -- no sobre las 71 completas.

Umbral de produccion: se usa app.main.fusion.umbral directamente (0.58),
NO la constante UMBRAL_PRODUCCION de validar_val.py (que quedo en 0.56,
desactualizada desde que main.py se recalibro a 0.58 -- ver
calibracion/experimento_adversarial.py, misma correccion).

Zona borderline: |score_comportamiento - umbral| <= ANCHO_BORDERLINE (0.05
por defecto -> banda 0.53-0.63 con umbral=0.58).

Paso 3 calcula, SOLO sobre las llamadas borderline con std_recuperacion
valido (excluye las que cayeron en eventos_insuficientes), el mismo
|diff medias| que usa calibrar.py (reportar_separacion_por_senal) y una
accuracy por barrido de umbral sobre std_recuperacion (ambas direcciones:
std alto = sospecha de sintetico, o std alto = sospecha de humano) --
calibrada y evaluada en el MISMO subconjunto borderline (n pequeño): es una
señal exploratoria, no una validacion held-out, y se reporta como tal.

Paso 4, solo si esa accuracy explora mejor que la regla actual (score>=
umbral) dentro del propio subconjunto borderline, simula una regla
combinada sobre las 71 completas: fuera de la zona borderline no cambia
nada (score>=umbral); dentro de la zona, si hay std_recuperacion valido,
decide con el umbral/direccion encontrados en el Paso 3; si no hay
std_recuperacion valido (eventos_insuficientes), cae de vuelta a
score>=umbral. Reporta si la accuracy total sobre las 71 mejora, empeora o
no cambia frente al baseline actual.

No modifica fusion.py ni main.py -- solo reporta numeros.

Uso tipico:
    python validar_defensa_borderline.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.main import cargar_canales, fusion

ANCHO_BORDERLINE = 0.05
RESULTADOS_VAL_POR_DEFECTO = Path(__file__).resolve().parent / "validacion_val_resultados.jsonl"
AUDIO_DIR_POR_DEFECTO = Path("/home/andres/hackmty26/audio")


def cargar_resultados_val(ruta: Path) -> list[dict]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"{ruta} no existe -- correr primero validar_val.py para generarlo."
        )
    registros = []
    for linea in ruta.read_text().splitlines():
        linea = linea.strip()
        if linea:
            registros.append(json.loads(linea))
    return registros


def reportar_borderline(registros: list[dict], umbral: float, ancho: float) -> list[dict]:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])
    pred = np.array([1 if r["score_comportamiento"] >= umbral else 0 for r in registros])
    accuracy_total = float(accuracy_score(y, pred))

    borderline = [r for r in registros if abs(r["score_comportamiento"] - umbral) <= ancho]

    print(f"=== Paso 1-2: zona borderline (umbral={umbral}, ancho=+/-{ancho}) ===")
    print(f"Llamadas en val: {len(registros)} — accuracy actual (score>=umbral): {accuracy_total:.4f}")
    print(
        f"Banda borderline: [{umbral - ancho:.2f}, {umbral + ancho:.2f}] — "
        f"{len(borderline)}/{len(registros)} llamadas caen ahi"
    )

    if borderline:
        y_b = np.array([1 if r["label"] == "synthetic" else 0 for r in borderline])
        pred_b = np.array([1 if r["score_comportamiento"] >= umbral else 0 for r in borderline])
        mal_clasificadas = int(np.sum(y_b != pred_b))
        print(f"De esas, mal clasificadas hoy (score>=umbral vs label real): {mal_clasificadas}/{len(borderline)}")

    return borderline


def recalcular_std_recuperacion(borderline: list[dict], audio_dir: Path) -> list[dict]:
    print("\n=== Recalculando std_recuperacion para el subconjunto borderline (no esta en el jsonl cacheado) ===")
    detector = DetectorComportamiento()
    con_std = []
    for r in borderline:
        anon_id = r["anon_id"]
        ruta_audio = audio_dir / f"{anon_id}.wav"
        if not ruta_audio.exists():
            print(f"  {anon_id}: audio no encontrado en {ruta_audio}, se omite")
            continue
        try:
            canal_caller, canal_callee, sr = cargar_canales(str(ruta_audio))
            senal = detector.analizar(canal_caller, canal_callee, sr)
        except Exception as exc:
            print(f"  {anon_id}: FALLO ({exc!r}), se omite")
            continue

        if abs(senal.score - r["score_comportamiento"]) > 1e-6:
            print(
                f"  {anon_id}: AVISO — score recalculado ({senal.score:.6f}) no coincide con "
                f"el cacheado ({r['score_comportamiento']:.6f}); revisar si el detector cambio."
            )

        std_recuperacion = senal.detalle.get("std_recuperacion")
        if std_recuperacion is None:
            print(f"  {anon_id}: sin std_recuperacion valido (eventos_insuficientes), se omite del Paso 3")
            continue

        con_std.append({
            "anon_id": anon_id,
            "label": r["label"],
            "score_comportamiento": r["score_comportamiento"],
            "std_recuperacion": std_recuperacion,
        })

    print(f"Llamadas borderline con std_recuperacion valido: {len(con_std)}/{len(borderline)}")
    return con_std


def evaluar_separacion_std(con_std: list[dict], umbral: float) -> dict | None:
    if len(con_std) < 2:
        print("\nMuy pocas llamadas borderline con std_recuperacion valido para medir separacion. Saliendo.")
        return None

    y = np.array([1 if r["label"] == "synthetic" else 0 for r in con_std])
    std = np.array([r["std_recuperacion"] for r in con_std])
    h, s = std[y == 0], std[y == 1]

    print(f"\n=== Paso 3: separacion de std_recuperacion, SOLO en borderline (n={len(con_std)}) ===")
    if len(h) == 0 or len(s) == 0:
        print("Falta alguna clase en el subconjunto borderline (todo human o todo synthetic) — no se puede medir.")
        return None

    diff_medias = abs(float(s.mean()) - float(h.mean()))
    print(
        f"std_recuperacion: humano mean={h.mean():.3f} std={h.std():.3f} | "
        f"sintetico mean={s.mean():.3f} std={s.std():.3f} | |diff medias|={diff_medias:.3f}"
    )

    pred_score = (np.array([r["score_comportamiento"] for r in con_std]) >= umbral).astype(int)
    accuracy_baseline_borderline = float(accuracy_score(y, pred_score))
    print(f"Accuracy baseline en este subconjunto (score>=umbral actual): {accuracy_baseline_borderline:.4f}")

    candidatos = np.linspace(std.min(), std.max(), 99)
    mejor_accuracy = -1.0
    mejor_umbral = None
    mejor_direccion = None
    for direccion in ("alto_es_sintetico", "alto_es_humano"):
        for t in candidatos:
            pred = (std >= t).astype(int) if direccion == "alto_es_sintetico" else (std <= t).astype(int)
            acc = accuracy_score(y, pred)
            if acc > mejor_accuracy:
                mejor_accuracy = float(acc)
                mejor_umbral = float(t)
                mejor_direccion = direccion

    print(
        f"Mejor accuracy con std_recuperacion solo (barrido de umbral, direccion={mejor_direccion}, "
        f"t={mejor_umbral:.3f}): {mejor_accuracy:.4f}"
    )
    print(
        "AVISO: este umbral/direccion se calibro Y evaluo en el mismo subconjunto borderline "
        "(n pequeño) — es una señal exploratoria de si hay algo que perseguir, no una validacion "
        "held-out. Un accuracy alto aqui puede ser sobreajuste al ruido de un n chico."
    )

    return {
        "diff_medias": diff_medias,
        "accuracy_baseline_borderline": accuracy_baseline_borderline,
        "mejor_accuracy_std": mejor_accuracy,
        "mejor_umbral_std": mejor_umbral,
        "mejor_direccion_std": mejor_direccion,
        "separa": mejor_accuracy > accuracy_baseline_borderline,
    }


def simular_regla_combinada(
    registros: list[dict],
    con_std: list[dict],
    umbral: float,
    ancho: float,
    umbral_std: float,
    direccion_std: str,
) -> None:
    std_por_anon_id = {r["anon_id"]: r["std_recuperacion"] for r in con_std}

    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])
    pred_baseline = np.array([1 if r["score_comportamiento"] >= umbral else 0 for r in registros])
    accuracy_baseline = float(accuracy_score(y, pred_baseline))

    pred_nueva = []
    for r in registros:
        en_borderline = abs(r["score_comportamiento"] - umbral) <= ancho
        std_recuperacion = std_por_anon_id.get(r["anon_id"])
        if en_borderline and std_recuperacion is not None:
            if direccion_std == "alto_es_sintetico":
                pred_nueva.append(1 if std_recuperacion >= umbral_std else 0)
            else:
                pred_nueva.append(1 if std_recuperacion <= umbral_std else 0)
        else:
            pred_nueva.append(1 if r["score_comportamiento"] >= umbral else 0)
    pred_nueva = np.array(pred_nueva)
    accuracy_nueva = float(accuracy_score(y, pred_nueva))

    print(f"\n=== Paso 4: regla combinada simulada sobre las {len(registros)} llamadas de val ===")
    print(f"Accuracy actual (solo score>=umbral):                {accuracy_baseline:.4f}")
    print(f"Accuracy con std_recuperacion como desempate borderline: {accuracy_nueva:.4f}")
    diff = accuracy_nueva - accuracy_baseline
    if diff > 1e-9:
        print(f"MEJORA: +{diff:.4f} ({diff * len(registros):.1f} llamadas netas)")
    elif diff < -1e-9:
        print(f"EMPEORA: {diff:.4f} ({diff * len(registros):.1f} llamadas netas)")
    else:
        print("SIN CAMBIO en accuracy total.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--resultados", default=str(RESULTADOS_VAL_POR_DEFECTO))
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR_POR_DEFECTO))
    parser.add_argument("--ancho-borderline", type=float, default=ANCHO_BORDERLINE)
    args = parser.parse_args()

    umbral = fusion.umbral
    registros = cargar_resultados_val(Path(args.resultados))

    borderline = reportar_borderline(registros, umbral, args.ancho_borderline)
    if not borderline:
        print("\nSin llamadas borderline. Nada que medir en los pasos 3-4.")
        return

    con_std = recalcular_std_recuperacion(borderline, Path(args.audio_dir))
    resultado_separacion = evaluar_separacion_std(con_std, umbral)
    if resultado_separacion is None:
        return

    if not resultado_separacion["separa"]:
        print(
            "\nstd_recuperacion NO mejora sobre la regla actual dentro del propio subconjunto "
            "borderline -- no vale la pena simular la regla combinada sobre las 71 completas."
        )
        return

    simular_regla_combinada(
        registros,
        con_std,
        umbral,
        args.ancho_borderline,
        resultado_separacion["mejor_umbral_std"],
        resultado_separacion["mejor_direccion_std"],
    )


if __name__ == "__main__":
    main()
