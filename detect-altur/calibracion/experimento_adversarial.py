"""experimento_adversarial.py — mide (NO arregla) el riesgo de un ataque de
timing simple contra DetectorComportamiento: como el score de produccion
depende UNICAMENTE de tiempo_primera_habla (ver
app/deteccion/comportamiento.py, _score_heuristico), un atacante que conozca
la media/std poblacional (MEDIA_TIEMPO_PRIMERA_HABLA=8.82s,
STD_TIEMPO_PRIMERA_HABLA=2.69s, ambas hardcodeadas en el codigo, no secretas)
podria intentar que su TTS/agente sintetico empiece a hablar justo en ese
punto para pasar por humano.

No se regenera audio: eso requeriria controlar el propio pipeline de sintesis
del atacante, que esta fuera de alcance. En su lugar se simula el resultado
del ataque de la forma mas directa posible: se toma el tiempo_primera_habla
REAL que el detector ya calculo para cada llamada synthetic de val, se
reemplaza por un valor objetivo cercano a la media humana, y se recalcula el
score con la MISMA formula de produccion (DetectorComportamiento.
_score_heuristico, sin reimplementarla) para ver si is_synthetic cambia de
True a False. Esto es el techo teorico del ataque (el atacante logra el
tiempo exacto que quiere); un ataque real sobre audio generado seria como
minimo igual de efectivo o peor (mas dificil de afinar con precision).

Valor objetivo elegido: exactamente MEDIA_TIEMPO_PRIMERA_HABLA (8.82s), el
centro de la banda "media +/- 1 std" pedida — el punto mas favorable posible
para el atacante (z-score=0, score exacto=0.5), y por construccion cae
dentro de esa banda.

No modifica fusion.py, main.py, comportamiento.py ni ningun peso/umbral --
solo lee resultados calculados y arma un ResultadoDeteccion nuevo con la
MISMA instancia de Fusion que usa produccion (app.main.fusion), para no
depender de constantes potencialmente desactualizadas en otros scripts de
calibracion (ver validar_val.UMBRAL_PRODUCCION, que quedo en 0.56 mientras
app/main.py ya usa 0.58 -- aqui se usa app.main.fusion directamente para
evitar ese riesgo).

Uso tipico:
    python experimento_adversarial.py
"""
import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.dominio.modelos import SenalScore
from app.main import cargar_canales, fusion
from validar_val import leer_manifest

MUESTRA = 10
SEMILLA = 0

# Punto objetivo del ataque: el atacante apunta exactamente a la media
# humana conocida (ver docstring del modulo).
TIEMPO_OBJETIVO_ATAQUE = DetectorComportamiento.MEDIA_TIEMPO_PRIMERA_HABLA


@dataclass
class Registro:
    anon_id: str
    tiempo_original: float
    tiempo_atacado: float
    score_original: float
    score_atacado: float
    is_synthetic_original: bool
    is_synthetic_atacado: bool


def muestrear_sinteticos(filas: list[dict], n: int, semilla: int) -> list[dict]:
    sinteticos = [f for f in filas if f["label"] == "synthetic"]
    rng = random.Random(semilla)
    rng.shuffle(sinteticos)
    return sinteticos[:n]


def evaluar_llamada(detector: DetectorComportamiento, ruta_audio: Path) -> Registro | None:
    canal_caller, canal_callee, sr = cargar_canales(str(ruta_audio))
    senal_original = detector.analizar(canal_caller, canal_callee, sr)

    tiempo_original = senal_original.detalle.get("tiempo_primera_habla")
    if tiempo_original is None:
        return None  # eventos_insuficientes u otro caso sin tiempo_primera_habla calculable

    score_atacado = detector._score_heuristico(TIEMPO_OBJETIVO_ATAQUE)
    senal_atacada = SenalScore(
        nombre="comportamiento",
        score=score_atacado,
        detalle={**senal_original.detalle, "tiempo_primera_habla": TIEMPO_OBJETIVO_ATAQUE},
    )

    resultado_original = fusion.combinar([senal_original])
    resultado_atacado = fusion.combinar([senal_atacada])

    return Registro(
        anon_id=ruta_audio.stem,
        tiempo_original=tiempo_original,
        tiempo_atacado=TIEMPO_OBJETIVO_ATAQUE,
        score_original=senal_original.score,
        score_atacado=score_atacado,
        is_synthetic_original=resultado_original.is_synthetic,
        is_synthetic_atacado=resultado_atacado.is_synthetic,
    )


def reportar_llamada(registro: Registro) -> None:
    flip = registro.is_synthetic_original and not registro.is_synthetic_atacado
    marca = " <-- EVADE deteccion" if flip else ""
    print(
        f"{registro.anon_id}: "
        f"tiempo_primera_habla {registro.tiempo_original:.2f}s -> {registro.tiempo_atacado:.2f}s | "
        f"score {registro.score_original:.3f} -> {registro.score_atacado:.3f} | "
        f"is_synthetic {registro.is_synthetic_original} -> {registro.is_synthetic_atacado}"
        f"{marca}"
    )


def reportar_resumen(registros: list[Registro]) -> None:
    n = len(registros)
    detectadas_original = sum(1 for r in registros if r.is_synthetic_original)
    flips = sum(1 for r in registros if r.is_synthetic_original and not r.is_synthetic_atacado)

    print(f"\n=== Resumen (n={n} llamadas synthetic de val, umbral={fusion.umbral}) ===")
    print(f"Detectadas correctamente ANTES del ataque: {detectadas_original}/{n}")
    print(
        f"De esas, dejarian de detectarse si el atacante fijara "
        f"tiempo_primera_habla={TIEMPO_OBJETIVO_ATAQUE:.2f}s (media humana): {flips}/{detectadas_original}"
    )
    print(f"Score resultante para TODAS las llamadas atacadas: {registros[0].score_atacado:.3f} (identico -- depende solo del tiempo objetivo, no de la llamada)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default="/home/andres/hackmty26/manifest.csv")
    parser.add_argument("--audio-dir", default="/home/andres/hackmty26/audio")
    parser.add_argument("--muestra", type=int, default=MUESTRA)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args()

    filas_val = leer_manifest(Path(args.manifest), split="val")
    muestra = muestrear_sinteticos(filas_val, args.muestra, args.semilla)
    print(f"Llamadas synthetic en split=val: {sum(1 for f in filas_val if f['label'] == 'synthetic')}")
    print(f"Muestra: {len(muestra)} llamadas synthetic (semilla={args.semilla})")
    print(
        f"Ataque simulado: forzar tiempo_primera_habla = {TIEMPO_OBJETIVO_ATAQUE:.2f}s "
        f"(media humana, banda +/-1std = [{DetectorComportamiento.MEDIA_TIEMPO_PRIMERA_HABLA - DetectorComportamiento.STD_TIEMPO_PRIMERA_HABLA:.2f}, "
        f"{DetectorComportamiento.MEDIA_TIEMPO_PRIMERA_HABLA + DetectorComportamiento.STD_TIEMPO_PRIMERA_HABLA:.2f}]s)\n"
    )

    detector = DetectorComportamiento()
    registros = []
    for fila in muestra:
        anon_id = fila["anon_id"]
        ruta_audio = Path(args.audio_dir) / f"{anon_id}.wav"
        if not ruta_audio.exists():
            print(f"{anon_id}: audio no encontrado en {ruta_audio}, se omite")
            continue
        try:
            registro = evaluar_llamada(detector, ruta_audio)
        except Exception as exc:
            print(f"{anon_id}: FALLO ({exc!r}), se omite")
            continue
        if registro is None:
            print(f"{anon_id}: sin tiempo_primera_habla calculable (eventos_insuficientes), se omite")
            continue
        registros.append(registro)
        reportar_llamada(registro)

    print(f"\nTotal de llamadas procesadas: {len(registros)}/{len(muestra)}")
    if not registros:
        print("Sin registros utilizables. Saliendo.")
        return

    reportar_resumen(registros)


if __name__ == "__main__":
    main()
