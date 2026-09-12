"""ataque_audio_real.py — version "de verdad" (no solo matematica) del
ataque de timing de experimento_adversarial.py: en vez de reemplazar el
numero tiempo_primera_habla y recalcular _score_heuristico, modifica el
WAV real (inserta/recorta silencio digital al inicio del canal caller con
numpy/soundfile) y vuelve a correr Silero VAD + DetectorComportamiento.
analizar() completo sobre el audio modificado, leido de disco de nuevo
(no se reusa el array en memoria) para ejercitar el mismo camino que
produccion.

Llamadas usadas: las 5 que experimento_adversarial.py (semilla=0, muestra=10
synthetic de val) marco "EVADE deteccion" -- is_synthetic paso de True a
False bajo el ataque matematico. Se corrio ese script para obtener esta
lista exacta (no se puede derivar solo de "tiempo_primera_habla fuera de
+/-1std [6.13, 11.51]s": experimento_adversarial.py marca EVADE cuando el
score ORIGINAL ya superaba el umbral de produccion (0.58), lo cual
corresponde a z >= 0.322, es decir tiempo_primera_habla >= 9.69s -- una
banda mas angosta que +/-1std. De hecho 2 de las 5 (call_4affad158a4c en
10.60s y call_810c24005ecc en 10.40s) caen DENTRO de [6.13, 11.51] y aun
asi evadieron, porque lo que importa es el umbral de produccion, no la
banda de +/-1std):

    call_4affad158a4c  (tiempo_primera_habla original 10.60s)
    call_e329faf2b6a1  (tiempo_primera_habla original 12.80s)
    call_af9ad367dcce  (tiempo_primera_habla original 12.10s)
    call_810c24005ecc  (tiempo_primera_habla original 10.40s)
    call_0847d7417bb1  (tiempo_primera_habla original 11.60s)

Todas synthetic (vienen del pool synthetic de val).

Metodo de manipulacion (solo canal caller, canal callee intacto, tal como
se pidio): se calcula delta = 8.82s (MEDIA_TIEMPO_PRIMERA_HABLA) menos el
tiempo_primera_habla real detectado por VAD en el audio ORIGINAL. Si
delta > 0 (hay que retrasar el habla), se inserta `delta` segundos de
silencio digital (ceros) al inicio del caller y se recorta la misma
cantidad de su final, para no cambiar el largo total del canal. Si
delta < 0 (hay que adelantarlo), se recorta `|delta|` segundos del inicio
y se rellena con silencio al final. Es una manipulacion simple (no reduce
ni fabrica habla real, solo desplaza el silencio existente) -- no se
toca el canal callee.

ADVERTENCIA que este script verifica explicitamente: _detectar_eventos()
usa solapamiento/silencios ENTRE caller y callee para contar
"recuperaciones"; si len(recuperaciones) < 2 el detector cae a
score=0.5 por la rama "eventos_insuficientes", SIN QUE ESO SIGNIFIQUE QUE
EL ATAQUE DE TIMING FUNCIONO -- podria ser que desplazar el caller sin
tocar el callee rompio la deteccion de eventos por completo. Este script
reporta, para cada llamada, si el score=0.5 real vino de
tiempo_primera_habla cerca de 8.82s con eventos normales, o de la rama
eventos_insuficientes (senal de que la manipulacion desincronizo los
canales en vez de "enganiar" limpiamente al detector).

No modifica fusion.py, main.py ni comportamiento.py -- solo genera WAVs de
prueba en --salida-dir (por defecto calibracion/audio_ataque_real/, fuera
del dataset real) y los analiza.

Uso tipico:
    python ataque_audio_real.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.dominio.modelos import SenalScore
from app.main import cargar_canales, fusion

AUDIO_DIR_POR_DEFECTO = Path("/home/andres/hackmty26/audio")
SALIDA_DIR_POR_DEFECTO = Path(__file__).resolve().parent / "audio_ataque_real"

TARGET_TIEMPO = DetectorComportamiento.MEDIA_TIEMPO_PRIMERA_HABLA  # 8.82s

# Las 5 llamadas que experimento_adversarial.py marco "EVADE deteccion"
# (ver docstring del modulo para la salida exacta que produjo esa lista).
LLAMADAS_EVADEN = (
    "call_4affad158a4c",
    "call_e329faf2b6a1",
    "call_af9ad367dcce",
    "call_810c24005ecc",
    "call_0847d7417bb1",
)


def _desplazar_silencio_array(caller: np.ndarray, sr: int, segundos_a_insertar: float) -> np.ndarray:
    """Retrasa (segundos_a_insertar>0) o adelanta (segundos_a_insertar<0) el
    contenido del canal insertando/recortando silencio digital al inicio,
    recortando/rellenando la misma cantidad al final para no cambiar el
    largo total (mantiene el canal caller alineado en longitud con el
    callee, que no se toca).
    """
    n_delta = min(int(round(abs(segundos_a_insertar) * sr)), len(caller))
    silencio = np.zeros(n_delta, dtype=caller.dtype)
    if segundos_a_insertar > 0:
        return np.concatenate([silencio, caller[: len(caller) - n_delta]])
    if segundos_a_insertar < 0:
        return np.concatenate([caller[n_delta:], silencio])
    return caller.copy()


def desplazar_silencio_inicio_wav(
    ruta_wav_entrada: Path, segundos_a_insertar: float, ruta_wav_salida: Path
) -> None:
    """Version reusable sobre archivos, no solo sobre las llamadas de val:
    lee ruta_wav_entrada (2 canales: caller=canal 0, callee=canal 1 -- si
    es mono, cargar_canales duplica el unico canal en ambos), inserta
    (segundos_a_insertar>0) o recorta (segundos_a_insertar<0) silencio
    digital al inicio del canal caller UNICAMENTE (callee intacto, ver
    _desplazar_silencio_array), y escribe el resultado en ruta_wav_salida.

    Sirve para aplicar el mismo ataque de timing a audio nuevo (p. ej. voz
    propia mezclada con un canal callee real), no solo a las llamadas de
    val ya anonimizadas.
    """
    canal_caller, canal_callee, sr = cargar_canales(str(ruta_wav_entrada))
    caller_modificado = _desplazar_silencio_array(canal_caller, sr, segundos_a_insertar)
    salida = np.stack([caller_modificado, canal_callee], axis=1)
    sf.write(ruta_wav_salida, salida, sr, subtype="PCM_16")


def tiempo_primera_habla_de(detector: DetectorComportamiento, caller: np.ndarray, sr: int) -> float | None:
    audio_caller = detector._resamplear(caller, sr)
    habla_caller = detector._timestamps_habla(audio_caller)
    return float(habla_caller[0]["start"]) if habla_caller else None


def procesar_llamada(
    detector: DetectorComportamiento, anon_id: str, audio_dir: Path, salida_dir: Path
) -> dict | None:
    ruta_original = audio_dir / f"{anon_id}.wav"
    if not ruta_original.exists():
        print(f"{anon_id}: audio no encontrado en {ruta_original}, se omite")
        return None

    canal_caller, _canal_callee, sr = cargar_canales(str(ruta_original))

    tiempo_original = tiempo_primera_habla_de(detector, canal_caller, sr)
    if tiempo_original is None:
        print(f"{anon_id}: VAD no detecto habla en el caller original, se omite")
        return None

    delta = TARGET_TIEMPO - tiempo_original
    ruta_salida = salida_dir / f"{anon_id}_ataque.wav"
    desplazar_silencio_inicio_wav(ruta_original, delta, ruta_salida)

    # Se relee el WAV recien escrito (no se reusa el array en memoria), para
    # correr el mismo camino que produccion sobre el audio real modificado.
    canal_caller_real, canal_callee_real, sr_real = cargar_canales(str(ruta_salida))
    senal_real = detector.analizar(canal_caller_real, canal_callee_real, sr_real)
    resultado_real = fusion.combinar([senal_real])

    senal_matematica = SenalScore(
        nombre="comportamiento",
        score=detector._score_heuristico(TARGET_TIEMPO),
        detalle={},
    )
    resultado_matematico = fusion.combinar([senal_matematica])

    return {
        "anon_id": anon_id,
        "tiempo_original": tiempo_original,
        "delta_aplicado": delta,
        "ruta_salida": ruta_salida,
        "tiempo_primera_habla_real_post_ataque": senal_real.detalle.get("tiempo_primera_habla"),
        "eventos_insuficientes_post_ataque": bool(senal_real.detalle.get("error") == "eventos_insuficientes"),
        "score_matematico": senal_matematica.score,
        "is_synthetic_matematico": resultado_matematico.is_synthetic,
        "score_real": senal_real.score,
        "is_synthetic_real": resultado_real.is_synthetic,
    }


def reportar_llamada(registro: dict) -> None:
    print(f"\n{registro['anon_id']}:")
    print(f"  tiempo_primera_habla original: {registro['tiempo_original']:.2f}s")
    print(f"  delta aplicado al caller: {registro['delta_aplicado']:+.2f}s -> {registro['ruta_salida']}")
    tph_real = registro["tiempo_primera_habla_real_post_ataque"]
    if tph_real is not None:
        print(f"  tiempo_primera_habla detectado por VAD en el audio modificado: {tph_real:.2f}s")
    else:
        print("  tiempo_primera_habla NO detectable en el audio modificado (eventos_insuficientes)")
    if registro["eventos_insuficientes_post_ataque"]:
        print(
            "  AVISO: score real=0.5 vino de la rama eventos_insuficientes (menos de 2 "
            "recuperaciones detectadas) -- el desplazamiento probablemente rompio la "
            "deteccion de eventos caller/callee, NO es evidencia de que el ataque de "
            "timing haya enganiado limpiamente al detector."
        )
    print(
        f"  simulado (numero):  score={registro['score_matematico']:.3f}  "
        f"is_synthetic={registro['is_synthetic_matematico']}"
    )
    print(
        f"  real (audio):       score={registro['score_real']:.3f}  "
        f"is_synthetic={registro['is_synthetic_real']}"
    )
    coincide = registro["is_synthetic_real"] == registro["is_synthetic_matematico"]
    print(f"  coincide con simulacion matematica: {'SI' if coincide else 'NO'}")


def reportar_resumen(registros: list[dict]) -> None:
    n = len(registros)
    evaden_real = sum(1 for r in registros if not r["is_synthetic_real"])
    coinciden = sum(1 for r in registros if r["is_synthetic_real"] == r["is_synthetic_matematico"])
    por_eventos_insuficientes = sum(1 for r in registros if r["eventos_insuficientes_post_ataque"])

    print(f"\n=== Resumen (n={n} llamadas) ===")
    print(f"Evaden deteccion con manipulacion REAL de audio: {evaden_real}/{n}")
    print(f"Coinciden con la simulacion matematica (mismo is_synthetic): {coinciden}/{n}")
    print(
        f"De las que evaden, cuantas lo hacen por eventos_insuficientes "
        f"(posible artefacto, no timing limpio): {por_eventos_insuficientes}/{n}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR_POR_DEFECTO))
    parser.add_argument("--salida-dir", default=str(SALIDA_DIR_POR_DEFECTO))
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    salida_dir = Path(args.salida_dir)
    salida_dir.mkdir(parents=True, exist_ok=True)

    print(f"Llamadas: {len(LLAMADAS_EVADEN)}  target tiempo_primera_habla={TARGET_TIEMPO:.2f}s")
    print(f"WAVs modificados se guardan en: {salida_dir}\n")

    detector = DetectorComportamiento()
    registros = []
    for anon_id in LLAMADAS_EVADEN:
        registro = procesar_llamada(detector, anon_id, audio_dir, salida_dir)
        if registro is not None:
            registros.append(registro)
            reportar_llamada(registro)

    print(f"\nTotal de llamadas procesadas: {len(registros)}/{len(LLAMADAS_EVADEN)}")
    if not registros:
        print("Sin registros utilizables. Saliendo.")
        return

    reportar_resumen(registros)


if __name__ == "__main__":
    main()
