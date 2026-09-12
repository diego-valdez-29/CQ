"""detectar_silencio_artificial.py — mide (NO implementa) si el silencio
insertado por ataque_audio_real.py (ceros digitales) es distinguible del
silencio real de linea telefonica que precede al primer habla en las
llamadas originales de val, via RMS simple.

Hipotesis a probar: ataque_audio_real.py inserta silencio digital PERFECTO
(np.zeros) al inicio del canal caller para desplazar tiempo_primera_habla
cerca de 8.82s (ver ese script). Ese silencio no tiene el ruido de fondo
real de una linea telefonica (line noise, hiss, etc.) que si esta presente
en el tramo de silencio real de las llamadas originales antes del primer
habla. Si esa diferencia es medible en RMS, seria una señal barata para
detectar ESTE ataque especifico -- no un chequeo general de "audio
sintetico".

Para cada llamada (5 originales de /home/andres/hackmty26/audio/ + sus 5
versiones atacadas en calibracion/audio_ataque_real/, generadas por
ataque_audio_real.py), se recalcula tiempo_primera_habla via Silero VAD
(mismo helper que ataque_audio_real.py, no se reimplementa) y se mide el
RMS del tramo del canal caller ANTES de ese timestamp (el "silencio previo
al primer habla" segun el propio detector).

No modifica fusion.py, main.py, comportamiento.py ni ataque_audio_real.py
-- solo lee WAVs ya generados y reporta numeros.

Requiere haber corrido antes ataque_audio_real.py (para que existan los 5
WAV atacados en --ataque-dir).

Uso tipico:
    python detectar_silencio_artificial.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.main import cargar_canales
from ataque_audio_real import LLAMADAS_EVADEN, tiempo_primera_habla_de

AUDIO_DIR_POR_DEFECTO = Path("/home/andres/hackmty26/audio")
ATAQUE_DIR_POR_DEFECTO = Path(__file__).resolve().parent / "audio_ataque_real"


def rms_silencio_previo(detector: DetectorComportamiento, ruta_wav: Path) -> dict | None:
    canal_caller, _canal_callee, sr = cargar_canales(str(ruta_wav))
    tiempo_primera_habla = tiempo_primera_habla_de(detector, canal_caller, sr)

    if tiempo_primera_habla is None:
        return None
    if tiempo_primera_habla <= 0:
        return {"tiempo_primera_habla": tiempo_primera_habla, "rms": None, "n_muestras": 0}

    n_muestras = int(round(tiempo_primera_habla * sr))
    segmento = canal_caller[:n_muestras]
    rms = float(np.sqrt(np.mean(segmento.astype(np.float64) ** 2)))
    return {"tiempo_primera_habla": tiempo_primera_habla, "rms": rms, "n_muestras": n_muestras}


def procesar_grupo(
    detector: DetectorComportamiento, anon_id: str, audio_dir: Path, ataque_dir: Path
) -> dict | None:
    ruta_original = audio_dir / f"{anon_id}.wav"
    ruta_atacada = ataque_dir / f"{anon_id}_ataque.wav"

    if not ruta_original.exists():
        print(f"{anon_id}: audio original no encontrado en {ruta_original}, se omite")
        return None
    if not ruta_atacada.exists():
        print(
            f"{anon_id}: audio atacado no encontrado en {ruta_atacada} "
            f"(¿corriste ataque_audio_real.py?), se omite"
        )
        return None

    original = rms_silencio_previo(detector, ruta_original)
    atacada = rms_silencio_previo(detector, ruta_atacada)

    if original is None or atacada is None:
        print(f"{anon_id}: VAD no detecto habla en original o atacada, se omite")
        return None

    return {"anon_id": anon_id, "original": original, "atacada": atacada}


def reportar_llamada(registro: dict) -> None:
    anon_id = registro["anon_id"]
    o, a = registro["original"], registro["atacada"]

    def fmt(r: dict) -> str:
        if r["rms"] is None:
            return f"tiempo_primera_habla={r['tiempo_primera_habla']:.2f}s (sin silencio previo, empieza a hablar desde el inicio)"
        return f"tiempo_primera_habla={r['tiempo_primera_habla']:.2f}s  RMS silencio previo={r['rms']:.8f} (n={r['n_muestras']} muestras)"

    print(f"\n{anon_id}:")
    print(f"  original: {fmt(o)}")
    print(f"  atacada : {fmt(a)}")


def proponer_umbral(registros: list[dict]) -> None:
    con_rms = [r for r in registros if r["original"]["rms"] is not None and r["atacada"]["rms"] is not None]
    print(f"\n=== Separacion RMS (n={len(con_rms)} pares con silencio previo medible) ===")

    if not con_rms:
        print("Ninguna llamada tiene silencio previo medible en ambas versiones. No hay nada que separar.")
        return

    rms_originales = np.array([r["original"]["rms"] for r in con_rms])
    rms_atacadas = np.array([r["atacada"]["rms"] for r in con_rms])

    print(f"RMS originales:  min={rms_originales.min():.8f}  max={rms_originales.max():.8f}  media={rms_originales.mean():.8f}")
    print(f"RMS atacadas:    min={rms_atacadas.min():.8f}  max={rms_atacadas.max():.8f}  media={rms_atacadas.mean():.8f}")

    separa_limpio = rms_atacadas.max() < rms_originales.min()
    if separa_limpio:
        umbral_propuesto = (rms_atacadas.max() + rms_originales.min()) / 2
        print(
            f"\nSeparacion limpia: todas las atacadas ({rms_atacadas.max():.8f} maximo) quedan "
            f"por debajo de todas las originales ({rms_originales.min():.8f} minimo)."
        )
        print(
            f"Umbral propuesto (punto medio): RMS < {umbral_propuesto:.8f} -> sospechoso de "
            f"silencio digital insertado. NO implementado en produccion, solo propuesto."
        )
    else:
        print(
            "\nNO hay separacion limpia por un solo umbral: el rango de RMS de atacadas "
            f"([{rms_atacadas.min():.8f}, {rms_atacadas.max():.8f}]) se solapa con el de "
            f"originales ([{rms_originales.min():.8f}, {rms_originales.max():.8f}]). Un "
            "umbral simple de RMS clasificaria mal al menos alguna de las 10 llamadas -- "
            "no se propone umbral."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR_POR_DEFECTO))
    parser.add_argument("--ataque-dir", default=str(ATAQUE_DIR_POR_DEFECTO))
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    ataque_dir = Path(args.ataque_dir)

    detector = DetectorComportamiento()
    registros = []
    for anon_id in LLAMADAS_EVADEN:
        registro = procesar_grupo(detector, anon_id, audio_dir, ataque_dir)
        if registro is not None:
            registros.append(registro)
            reportar_llamada(registro)

    print(f"\nTotal de pares procesados: {len(registros)}/{len(LLAMADAS_EVADEN)}")
    if not registros:
        print("Sin registros utilizables. Saliendo.")
        return

    proponer_umbral(registros)


if __name__ == "__main__":
    main()
