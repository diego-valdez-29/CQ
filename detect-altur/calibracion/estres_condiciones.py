"""estres_condiciones.py — prueba de robustez (NO calibracion) del pipeline de
produccion (solo DetectorComportamiento + Fusion, pesos ya fijos) ante 3
condiciones de audio degradado que el dataset de calibracion no cubre:
ruido de linea, codec agresivo y sub-muestreo extra.

Toma una muestra balanceada de 10 llamadas de split="val" (5 human / 5
synthetic, reutilizando leer_manifest/muestrear_balanceado de calibrar.py) y,
para cada una, genera 3 versiones degradadas del WAV original:

  - ruido:             ruido blanco anadido solo al canal caller, SNR~20dB
                        (calculado en numpy porque el SNR objetivo depende de
                        la potencia real de la senal de cada llamada; no es
                        algo que ffmpeg calcule por si solo).
  - codec_agresivo:     pasa el audio (ambos canales) por mp3 a 8kbps y de
                        vuelta a wav via ffmpeg (subprocess), simulando una
                        linea telefonica mala.
  - downsample_extra:   reduce el sample rate a 4000Hz y lo vuelve a subir a
                        8000Hz via ffmpeg (subprocess), simulando peor
                        calidad de captura.

Corre exactamente el mismo camino que /detect en produccion sobre cada
version (original + 3 degradadas): reutiliza DetectorComportamiento,
Fusion y cargar_canales sin modificarlos, y reutiliza la funcion fusionar()
de validar_val.py (misma fusion, mismos pesos={comportamiento:1.0},
umbral=0.56, ya fijados en produccion).

Reporta por llamada el score/is_synthetic en cada version y si cambio contra
el original, y al final cuantas de las 10 llamadas cambiaron de
clasificacion bajo cada tipo de degradacion — esa cuenta es la metrica de
robustez ante condiciones no vistas en calibracion.

No modifica fusion.py, main.py ni ningun peso.

Uso tipico:
    python estres_condiciones.py
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Este script vive en calibracion/, pero "app" esta un nivel arriba (raiz del
# proyecto). Python solo agrega el directorio del propio script a sys.path,
# no el cwd, asi que sin esto el import de abajo falla al correrlo desde
# fuera de calibracion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.deteccion.comportamiento import DetectorComportamiento
from app.main import cargar_canales
from calibrar import leer_manifest, muestrear_balanceado
from validar_val import fusionar, PESOS_PRODUCCION, UMBRAL_PRODUCCION

MUESTRA = 10
SEMILLA = 0
SNR_DB = 20.0
NOMBRES_DEGRADACIONES = ("ruido", "codec_agresivo", "downsample_extra")


def _correr_ffmpeg(args: list[str]) -> None:
    resultado = subprocess.run(args, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(f"ffmpeg fallo ({' '.join(args)}):\n{resultado.stderr}")


def degradar_ruido(ruta_entrada: Path, ruta_salida: Path, anon_id: str, snr_db: float = SNR_DB) -> None:
    """Anade ruido blanco SOLO al canal caller (canal 0), con la amplitud
    necesaria para lograr el SNR objetivo respecto a la potencia real de esa
    senal (por eso se calcula en numpy y no con un filtro generico de
    ffmpeg: el SNR depende de la RMS de cada llamada, que no se conoce de
    antemano). El canal callee queda intacto.
    """
    data, sr = sf.read(ruta_entrada, always_2d=True)
    caller = data[:, 0].astype(np.float64)
    callee = data[:, 1].astype(np.float64) if data.shape[1] > 1 else caller.copy()

    rms_senal = np.sqrt(np.mean(caller ** 2))
    rms_ruido = rms_senal / (10 ** (snr_db / 20))

    semilla = int(hashlib.sha256(anon_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(0.0, rms_ruido, size=caller.shape)

    caller_con_ruido = np.clip(caller + ruido, -32768, 32767)
    salida = np.stack([caller_con_ruido, callee], axis=1).astype(np.int16)
    sf.write(ruta_salida, salida, sr, subtype="PCM_16")


def degradar_codec_agresivo(ruta_entrada: Path, ruta_salida: Path) -> None:
    """mp3 a 8kbps y de vuelta a wav (ambos canales), via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_mp3 = Path(tmpdir) / "tmp.mp3"
        _correr_ffmpeg(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta_entrada), "-b:a", "8k", str(ruta_mp3)]
        )
        _correr_ffmpeg(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta_mp3), str(ruta_salida)]
        )


def degradar_downsample_extra(ruta_entrada: Path, ruta_salida: Path) -> None:
    """4000Hz y de vuelta a 8000Hz (ambos canales), via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_baja = Path(tmpdir) / "tmp_4k.wav"
        _correr_ffmpeg(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta_entrada), "-ar", "4000", str(ruta_baja)]
        )
        _correr_ffmpeg(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta_baja), "-ar", "8000", str(ruta_salida)]
        )


def generar_version(nombre: str, ruta_original: Path, ruta_salida: Path, anon_id: str) -> None:
    if nombre == "ruido":
        degradar_ruido(ruta_original, ruta_salida, anon_id)
    elif nombre == "codec_agresivo":
        degradar_codec_agresivo(ruta_original, ruta_salida)
    elif nombre == "downsample_extra":
        degradar_downsample_extra(ruta_original, ruta_salida)
    else:
        raise ValueError(f"degradacion desconocida: {nombre}")


def evaluar_version(detector: DetectorComportamiento, ruta: Path) -> dict:
    canal_caller, canal_callee, sr = cargar_canales(str(ruta))
    senal = detector.analizar(canal_caller, canal_callee, sr)
    resultado = fusionar(senal.score)
    return {"score": senal.score, "confidence": resultado.confidence, "is_synthetic": resultado.is_synthetic}


def procesar_llamada(detector: DetectorComportamiento, fila: dict, audio_dir: Path) -> dict | None:
    anon_id = fila["anon_id"]
    ruta_original = audio_dir / f"{anon_id}.wav"
    if not ruta_original.exists():
        print(f"  {anon_id}: audio no encontrado en {ruta_original}, se omite")
        return None

    resultados = {}
    try:
        resultados["original"] = evaluar_version(detector, ruta_original)
        with tempfile.TemporaryDirectory() as tmpdir:
            for nombre in NOMBRES_DEGRADACIONES:
                ruta_degradada = Path(tmpdir) / f"{nombre}.wav"
                generar_version(nombre, ruta_original, ruta_degradada, anon_id)
                resultados[nombre] = evaluar_version(detector, ruta_degradada)
    except Exception as exc:
        print(f"  {anon_id}: FALLO ({exc!r}), se omite")
        return None

    return {"anon_id": anon_id, "label": fila["label"], "resultados": resultados}


def reportar_llamada(registro: dict) -> None:
    anon_id, label, resultados = registro["anon_id"], registro["label"], registro["resultados"]
    original = resultados["original"]
    print(f"\n{anon_id} (label={label}):")
    print(f"  original          : score={original['score']:.3f} is_synthetic={original['is_synthetic']}")
    for nombre in NOMBRES_DEGRADACIONES:
        r = resultados[nombre]
        cambio = "CAMBIO" if r["is_synthetic"] != original["is_synthetic"] else "igual"
        print(
            f"  {nombre:<18}: score={r['score']:.3f} is_synthetic={r['is_synthetic']} ({cambio} vs original)"
        )


def reportar_resumen(registros: list[dict]) -> None:
    n = len(registros)
    print(f"\n=== Resumen de robustez (n={n} llamadas, pesos={PESOS_PRODUCCION} umbral={UMBRAL_PRODUCCION}) ===")
    for nombre in NOMBRES_DEGRADACIONES:
        cambios = sum(
            1 for r in registros if r["resultados"][nombre]["is_synthetic"] != r["resultados"]["original"]["is_synthetic"]
        )
        print(f"  {nombre:<18}: {cambios}/{n} llamadas cambiaron de clasificacion")


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
    print(f"Llamadas en split=val: {len(filas_val)}")

    muestra = muestrear_balanceado(filas_val, args.muestra, args.semilla)
    print(f"Muestra balanceada: {len(muestra)} llamadas ({args.muestra // 2} human / {args.muestra // 2} synthetic)")

    detector = DetectorComportamiento()
    registros = []
    for fila in muestra:
        registro = procesar_llamada(detector, fila, Path(args.audio_dir))
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
