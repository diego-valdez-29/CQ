"""analisis_falsos_positivos.py — analiza (NO cambia nada) el patron de
error en las llamadas human de val que HOY se clasifican como synthetic
(falsos positivos), para entender que tan cerca del umbral estan en
terminos de tiempo_primera_habla, no solo de score.

Umbral usado: app.main.fusion.umbral (0.58, el real de produccion), NO la
constante UMBRAL_PRODUCCION de validar_val.py (que quedo en 0.56,
desactualizada -- ver el mismo aviso en experimento_adversarial.py y
validar_defensa_borderline.py). Esto importa aqui en particular: a
umbral=0.58 hay 6 falsos positivos human ([[31,6],[9,25]] TN,FP,FN,TP), NO
10 -- los 10 (matriz [[27,10],[9,25]]) son los falsos positivos a
umbral=0.56, el umbral viejo que ya no esta en produccion. Verificado
corriendo ambos umbrales sobre validacion_val_resultados.jsonl antes de
escribir este script.

DetectorComportamiento._score_heuristico(tiempo_primera_habla) es un
sigmoide de un z-score 1D (ver app/deteccion/comportamiento.py), asi que
es invertible en forma cerrada: existe un tiempo_primera_habla exacto
("tiempo_umbral") que produce score == umbral. Para cada falso positivo se
reporta tiempo_primera_habla real (recalculado via VAD, no esta en el
jsonl cacheado -- ver el mismo patron en validar_defensa_borderline.py) y
tiempo_primera_habla real - tiempo_umbral: los segundos que le sobran por
encima del umbral (cuanto tendria que haber tardado MENOS en hablar para
clasificarse como human).

No modifica fusion.py, main.py ni comportamiento.py -- solo lee scores
cacheados + recalcula tiempo_primera_habla para el subconjunto pequeño de
falsos positivos, y reporta numeros.

Uso tipico:
    python analisis_falsos_positivos.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.main import cargar_canales, fusion
from ataque_audio_real import tiempo_primera_habla_de

RESULTADOS_VAL_POR_DEFECTO = Path(__file__).resolve().parent / "validacion_val_resultados.jsonl"
AUDIO_DIR_POR_DEFECTO = Path("/home/andres/hackmty26/audio")


def cargar_resultados_val(ruta: Path) -> list[dict]:
    if not ruta.exists():
        raise FileNotFoundError(f"{ruta} no existe -- correr primero validar_val.py para generarlo.")
    registros = []
    for linea in ruta.read_text().splitlines():
        linea = linea.strip()
        if linea:
            registros.append(json.loads(linea))
    return registros


def tiempo_umbral(detector: DetectorComportamiento, umbral: float) -> float:
    """Inversa cerrada de _score_heuristico: el tiempo_primera_habla exacto
    que produce score == umbral (sigmoide de un z-score 1D, invertible via
    logit). No reimplementa el sigmoide -- solo lo invierte algebraicamente
    usando la misma media/std que usa el detector.
    """
    z_umbral = float(np.log(umbral / (1 - umbral)))
    return detector._media_tiempo_primera_habla + z_umbral * detector._std_tiempo_primera_habla


def identificar_falsos_positivos(registros: list[dict], umbral: float) -> list[dict]:
    return [
        r for r in registros
        if r["label"] == "human" and r["score_comportamiento"] >= umbral
    ]


def procesar_falso_positivo(
    detector: DetectorComportamiento, registro: dict, audio_dir: Path, t_umbral: float
) -> dict | None:
    anon_id = registro["anon_id"]
    ruta_audio = audio_dir / f"{anon_id}.wav"
    if not ruta_audio.exists():
        print(f"  {anon_id}: audio no encontrado en {ruta_audio}, se omite")
        return None

    canal_caller, _canal_callee, sr = cargar_canales(str(ruta_audio))
    tiempo_real = tiempo_primera_habla_de(detector, canal_caller, sr)
    if tiempo_real is None:
        print(f"  {anon_id}: VAD no detecto habla en el caller, se omite (inesperado para un falso positivo)")
        return None

    score_recalculado = detector._score_heuristico(tiempo_real)
    if abs(score_recalculado - registro["score_comportamiento"]) > 1e-6:
        print(
            f"  {anon_id}: AVISO — score recalculado ({score_recalculado:.6f}) no coincide con "
            f"el cacheado ({registro['score_comportamiento']:.6f})"
        )

    return {
        "anon_id": anon_id,
        "score": registro["score_comportamiento"],
        "tiempo_primera_habla": tiempo_real,
        "segundos_sobre_umbral": tiempo_real - t_umbral,
    }


def reportar_falsos_positivos(registros: list[dict], t_umbral: float) -> None:
    print(f"\ntiempo_primera_habla en el umbral (score == umbral): {t_umbral:.3f}s\n")
    for r in sorted(registros, key=lambda r: r["segundos_sobre_umbral"]):
        print(
            f"  {r['anon_id']}: score={r['score']:.3f}  tiempo_primera_habla={r['tiempo_primera_habla']:.2f}s  "
            f"segundos sobre el umbral={r['segundos_sobre_umbral']:+.2f}s"
        )


def reportar_patron(registros: list[dict]) -> None:
    n = len(registros)
    distancias = np.array([r["segundos_sobre_umbral"] for r in registros])

    print(f"\n=== Patron entre los {n} falsos positivos ===")
    print(f"segundos sobre el umbral: min={distancias.min():.2f}  max={distancias.max():.2f}  "
          f"media={distancias.mean():.2f}  mediana={np.median(distancias):.2f}  std={distancias.std():.2f}")

    CERCA_S = 1.0
    cerca = [r for r in registros if r["segundos_sobre_umbral"] <= CERCA_S]
    lejos = [r for r in registros if r["segundos_sobre_umbral"] > CERCA_S]

    print(f"\nA <= {CERCA_S:.1f}s del umbral (borderline, un VAD ligeramente distinto los reclasificaria): {len(cerca)}/{n}")
    for r in cerca:
        print(f"  {r['anon_id']}: +{r['segundos_sobre_umbral']:.2f}s")

    print(f"\nA > {CERCA_S:.1f}s del umbral (outliers, tardaron bastante mas que un humano tipico): {len(lejos)}/{n}")
    for r in lejos:
        print(f"  {r['anon_id']}: +{r['segundos_sobre_umbral']:.2f}s")

    if not lejos:
        print(
            "\nTodos los falsos positivos estan cerca del umbral -- consistente con ruido de "
            "medicion/VAD en el borde de la decision, no con un patron sistematico distinto."
        )
    elif not cerca:
        print(
            "\nTodos los falsos positivos son outliers lejos del umbral -- sugiere una subpoblacion "
            "de humanos que genuinamente tardan mucho en hablar por primera vez (no ruido de borde)."
        )
    else:
        print(
            "\nMezcla de casos borderline y outliers -- no hay un patron unico; valdria la pena "
            "revisar los outliers individualmente (¿llamadas con ruido de fondo largo al inicio? "
            "¿silencios reales largos del caller?)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--resultados", default=str(RESULTADOS_VAL_POR_DEFECTO))
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR_POR_DEFECTO))
    args = parser.parse_args()

    umbral = fusion.umbral
    registros = cargar_resultados_val(Path(args.resultados))
    print(f"Llamadas en val: {len(registros)}  umbral de produccion: {umbral}")

    falsos_positivos_crudos = identificar_falsos_positivos(registros, umbral)
    print(f"Falsos positivos human (score >= {umbral}): {len(falsos_positivos_crudos)}/{len(registros)}")
    if not falsos_positivos_crudos:
        print("Sin falsos positivos. Saliendo.")
        return

    detector = DetectorComportamiento()
    t_umbral = tiempo_umbral(detector, umbral)

    print(f"\nRecalculando tiempo_primera_habla real para cada falso positivo (no esta en el jsonl cacheado)...")
    resultados = []
    for r in falsos_positivos_crudos:
        resultado = procesar_falso_positivo(detector, r, Path(args.audio_dir), t_umbral)
        if resultado is not None:
            resultados.append(resultado)

    print(f"\nTotal procesados: {len(resultados)}/{len(falsos_positivos_crudos)}")
    if not resultados:
        print("Sin registros utilizables. Saliendo.")
        return

    reportar_falsos_positivos(resultados, t_umbral)
    reportar_patron(resultados)


if __name__ == "__main__":
    main()
