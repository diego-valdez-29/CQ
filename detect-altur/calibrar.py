"""calibrar.py — calibra los pesos de Fusion y valida la decision secuencial
contra el dataset real de Altur (~/hackmty26).

IMPORTANTE: para que la señal "semantico" use el LLM real (no su fallback
score=0.5) este script debe correrse en una maquina con acceso por Tailscale
a la Spark (SPARK_API_KEY debe estar seteada en el entorno). Sin eso, corre
igual pero la calibracion de esa señal no sirve de nada.

Uso tipico:
    pip install -r requirements-calibracion.txt
    export SPARK_API_KEY=...
    python calibrar.py                    # muestra balanceada de 80 llamadas
    python calibrar.py --muestra 60
    python calibrar.py --todas            # las 282 llamadas de train completas

Guarda resultados crudos incrementalmente en --salida (jsonl) y puede
reanudarse si se interrumpe: al volver a correr, salta las llamadas ya
procesadas en ese archivo (usa --reiniciar para ignorarlas y empezar de cero).
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

from app.deteccion.acustico import DetectorAcustico
from app.deteccion.comportamiento import DetectorComportamiento
from app.deteccion.fusion import Fusion
from app.deteccion.semantico import DetectorSemantico
from app.dominio.modelos import SenalScore
from app.main import CHECKPOINTS_S, UMBRAL_DECISION_TEMPRANA, cargar_canales

NOMBRES_SENALES = ("acustico", "comportamiento", "semantico")
PESOS_ACTUALES = {"acustico": 1.0, "comportamiento": 1.0, "semantico": 1.0}
UMBRAL_ACTUAL = 0.5

MUESTRA_POR_DEFECTO = 80


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


def truncar(canal_caller: np.ndarray, canal_callee: np.ndarray, sr: int, hasta_s: float):
    n = int(hasta_s * sr)
    return canal_caller[:n], canal_callee[:n]


def correr_detectores(detectores, canal_caller: np.ndarray, canal_callee: np.ndarray, sr: int) -> dict:
    return {
        senal.nombre: senal.score
        for senal in (d.analizar(canal_caller, canal_callee, sr) for d in detectores)
    }


def calcular_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def fusionar(scores: dict, pesos: dict, umbral: float):
    senales = [SenalScore(nombre=n, score=scores[n], detalle={}) for n in NOMBRES_SENALES]
    return Fusion(pesos=pesos, umbral=umbral).combinar(senales)


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

    detectores = [DetectorAcustico(), DetectorComportamiento(), DetectorSemantico()]
    modo_escritura = "a" if (registros and not reiniciar) else "w"
    archivo_salida = open(ruta_salida, modo_escritura)

    t_inicio = time.time()
    for i, fila in enumerate(pendientes, start=1):
        anon_id = fila["anon_id"]
        ruta_audio = audio_dir / f"{anon_id}.wav"
        if not ruta_audio.exists():
            print(f"  [{i}/{len(pendientes)}] {anon_id}: audio no encontrado en {ruta_audio}, se omite")
            continue

        t0 = time.time()
        try:
            canal_caller, canal_callee, sr = cargar_canales(str(ruta_audio))
            duracion_s = len(canal_caller) / sr

            scores_full = correr_detectores(detectores, canal_caller, canal_callee, sr)

            scores_checkpoint = {}
            for checkpoint_s in CHECKPOINTS_S:
                if duracion_s <= checkpoint_s:
                    continue
                c_caller, c_callee = truncar(canal_caller, canal_callee, sr, checkpoint_s)
                scores_checkpoint[f"{checkpoint_s:.0f}s"] = correr_detectores(
                    detectores, c_caller, c_callee, sr
                )
        except Exception as exc:
            print(f"  [{i}/{len(pendientes)}] {anon_id}: FALLO ({exc!r}), se omite")
            continue

        registro = {
            "anon_id": anon_id,
            "label": fila["label"],
            "duracion_s": duracion_s,
            "scores_full": scores_full,
            "scores_checkpoint": scores_checkpoint,
        }
        registros.append(registro)
        archivo_salida.write(json.dumps(registro) + "\n")
        archivo_salida.flush()

        elapsed = time.time() - t0
        promedio = (time.time() - t_inicio) / i
        restante_s = promedio * (len(pendientes) - i)
        print(
            f"  [{i}/{len(pendientes)}] {anon_id} ({fila['label']}, {duracion_s:.0f}s) "
            f"scores={scores_full} ({elapsed:.1f}s, ~{restante_s / 60:.1f}min restantes)"
        )

    archivo_salida.close()
    return registros


def reportar_separacion_por_senal(registros: list[dict]) -> None:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])
    print("\n=== Paso 3: separacion por señal individual (score full-call) ===")
    for nombre in NOMBRES_SENALES:
        scores = np.array([r["scores_full"][nombre] for r in registros])
        h, s = scores[y == 0], scores[y == 1]
        print(
            f"{nombre}: humano mean={h.mean():.3f} std={h.std():.3f} | "
            f"sintetico mean={s.mean():.3f} std={s.std():.3f} | "
            f"|diff medias|={abs(s.mean() - h.mean()):.3f}"
        )


def calibrar_pesos(registros: list[dict]) -> tuple[dict, float]:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])
    X = np.array([[r["scores_full"][n] for n in NOMBRES_SENALES] for r in registros])

    modelo = LogisticRegression()
    modelo.fit(X, y)
    coef_crudo = modelo.coef_[0]

    negativos = [n for n, c in zip(NOMBRES_SENALES, coef_crudo) if c < 0]
    if negativos:
        print(
            f"AVISO: coeficiente negativo en {negativos} — la regresion encontro que "
            "un score mas alto en esa señal correlaciona con 'humano', al reves de la "
            "convencion (score->1 = sintetico). Se pone su peso en 0 (no se confia en "
            "ella) en vez de tomar valor absoluto."
        )

    coef = np.clip(coef_crudo, 0, None)
    if coef.sum() == 0:
        print(
            "AVISO: las tres señales salieron con coeficiente negativo; como peso 0 en "
            "las tres dejaria la fusion indefinida, se usa abs(coeficiente) de las tres "
            "como fallback en vez de 0."
        )
        coef = np.abs(coef_crudo)
    pesos_calibrados = {n: float(c / coef.sum()) for n, c in zip(NOMBRES_SENALES, coef)}

    confidences = np.array(
        [fusionar(r["scores_full"], pesos_calibrados, umbral=0.5).confidence for r in registros]
    )
    candidatos = np.linspace(0.01, 0.99, 99)
    accuracies = [accuracy_score(y, confidences >= u) for u in candidatos]
    umbral_calibrado = float(candidatos[int(np.argmax(accuracies))])

    print("\n=== Paso 4: pesos calibrados (regresion logistica) ===")
    print("Coeficientes crudos de la regresion:", dict(zip(NOMBRES_SENALES, modelo.coef_[0].tolist())))
    print("Pesos normalizados para Fusion:", pesos_calibrados)
    print("Umbral calibrado (max accuracy en el propio set de calibracion):", umbral_calibrado)
    return pesos_calibrados, umbral_calibrado


def reportar_comparacion(registros: list[dict], pesos_calibrados: dict, umbral_calibrado: float) -> None:
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in registros])

    for titulo, pesos, umbral in (
        ("PESOS ACTUALES (placeholder 1/1/1, umbral 0.5)", PESOS_ACTUALES, UMBRAL_ACTUAL),
        ("PESOS CALIBRADOS (regresion logistica)", pesos_calibrados, umbral_calibrado),
    ):
        resultados = [fusionar(r["scores_full"], pesos, umbral) for r in registros]
        pred = np.array([r.is_synthetic for r in resultados], dtype=int)
        conf = np.array([r.confidence for r in resultados])

        print(f"\n=== Paso 5: {titulo} ===")
        print("Accuracy:", accuracy_score(y, pred))
        print("Matriz de confusion [[TN,FP],[FN,TP]]:\n", confusion_matrix(y, pred))
        print("EER:", calcular_eer(y, conf))


def reportar_decision_temprana(registros: list[dict], pesos_calibrados: dict) -> None:
    print(
        f"\n=== Paso 6: decision temprana con pesos calibrados "
        f"(umbral_decision_temprana={UMBRAL_DECISION_TEMPRANA}) ==="
    )
    n_decididas = 0
    desglose = {f"{c:.0f}s": 0 for c in CHECKPOINTS_S}
    n_sin_checkpoint_valido = 0

    for r in registros:
        decidido = False
        for checkpoint_s in CHECKPOINTS_S:
            clave = f"{checkpoint_s:.0f}s"
            scores_cp = r["scores_checkpoint"].get(clave)
            if scores_cp is None:
                continue
            resultado_cp = fusionar(scores_cp, pesos_calibrados, umbral=UMBRAL_DECISION_TEMPRANA)
            if resultado_cp.is_synthetic and resultado_cp.confidence >= UMBRAL_DECISION_TEMPRANA:
                n_decididas += 1
                desglose[clave] += 1
                decidido = True
                break
        if not decidido and not any(
            f"{c:.0f}s" in r["scores_checkpoint"] for c in CHECKPOINTS_S
        ):
            n_sin_checkpoint_valido += 1

    total = len(registros)
    pct = 100 * n_decididas / total if total else 0.0
    print(
        f"{n_decididas}/{total} llamadas ({pct:.1f}%) hubieran cruzado el umbral de "
        f"decision temprana en el checkpoint de 25s o 40s."
    )
    print("Desglose por checkpoint:", desglose)
    if n_sin_checkpoint_valido:
        print(
            f"({n_sin_checkpoint_valido} llamadas eran mas cortas que ambos checkpoints "
            "y nunca tuvieron oportunidad de decidir temprano.)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default="/home/andres/hackmty26/manifest.csv")
    parser.add_argument("--audio-dir", default="/home/andres/hackmty26/audio")
    parser.add_argument("--salida", default="calibracion_resultados.jsonl")
    parser.add_argument(
        "--muestra",
        type=int,
        default=MUESTRA_POR_DEFECTO,
        help=f"tamano de muestra balanceada por clase (default {MUESTRA_POR_DEFECTO})",
    )
    parser.add_argument("--todas", action="store_true", help="usar las 282 llamadas de train completas")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument(
        "--reiniciar", action="store_true", help="ignora --salida existente y empieza de cero"
    )
    args = parser.parse_args()

    filas_train = leer_manifest(Path(args.manifest), split="train")
    print(f"Llamadas en split=train: {len(filas_train)}")

    if args.todas:
        filas = filas_train
    else:
        filas = muestrear_balanceado(filas_train, args.muestra, args.semilla)
        print(
            f"MUESTRA: usando {len(filas)} llamadas balanceadas por clase "
            f"(no las {len(filas_train)} completas de train) porque no se paso --todas."
        )

    registros = recolectar_datos(filas, Path(args.audio_dir), Path(args.salida), args.reiniciar)
    print(f"\nTotal de llamadas con datos utilizables: {len(registros)}/{len(filas)}")

    if len(registros) < 10:
        print("Muy pocos registros utilizables para calibrar de forma confiable. Saliendo.")
        return

    reportar_separacion_por_senal(registros)
    pesos_calibrados, umbral_calibrado = calibrar_pesos(registros)
    reportar_comparacion(registros, pesos_calibrados, umbral_calibrado)
    reportar_decision_temprana(registros, pesos_calibrados)


if __name__ == "__main__":
    main()
