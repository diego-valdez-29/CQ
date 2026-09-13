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

Cada WAV degradado se guarda en --audio-salida-dir (por defecto
calibracion/audio_estres/<anon_id>_<tipo>.wav, p.ej.
call_0e1e2f29bfdc_ruido.wav) en vez de un directorio temporal descartable,
para poder reusarlos como demo en el frontend. No cambia como se generan
ni se evaluan, solo donde quedan.

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
AUDIO_SALIDA_DIR_POR_DEFECTO = Path(__file__).resolve().parent / "audio_estres"


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


def verificar_sanity(ruta_original: Path, anon_id: str) -> None:
    """Chequeo previo a confiar en el resultado de robustez: confirma que
    codec_agresivo y downsample_extra realmente modifican el audio (y no
    devuelven una copia identica del original por algun error de ffmpeg,
    ruta mal armada, etc). Compara bytes crudos del WAV y, si difieren,
    tambien la RMS (potencia) de la senal para verificar que el cambio es
    audible/medible y no solo un header distinto.
    """
    print(f"\n=== Verificacion de sanity (la degradacion se aplica de verdad?) ===")
    print(f"Llamada de prueba: {anon_id} ({ruta_original})")

    bytes_original = ruta_original.read_bytes()
    data_orig, sr_orig = sf.read(ruta_original, always_2d=True)
    rms_orig = np.sqrt(np.mean(data_orig.astype(np.float64) ** 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        for nombre in ("codec_agresivo", "downsample_extra"):
            ruta_degradada = Path(tmpdir) / f"{nombre}.wav"
            generar_version(nombre, ruta_original, ruta_degradada, anon_id)
            bytes_degradada = ruta_degradada.read_bytes()
            identicos = bytes_original == bytes_degradada

            print(f"\n  {nombre}:")
            print(f"    bytes identicos al original: {identicos}")
            print(f"    tamano original: {len(bytes_original)} bytes, degradado: {len(bytes_degradada)} bytes")

            if identicos:
                print("    ALERTA: el WAV degradado es byte-a-byte igual al original -- la degradacion NO se aplico")
                continue

            data_deg, sr_deg = sf.read(ruta_degradada, always_2d=True)
            n = min(data_orig.shape[0], data_deg.shape[0])
            rms_deg = np.sqrt(np.mean(data_deg[:n].astype(np.float64) ** 2))
            rms_orig_n = np.sqrt(np.mean(data_orig[:n].astype(np.float64) ** 2))
            diff_rel = (rms_deg - rms_orig_n) / rms_orig_n if rms_orig_n > 0 else float("nan")
            db = 20 * np.log10(rms_deg / rms_orig_n) if rms_orig_n > 0 and rms_deg > 0 else float("nan")
            print(f"    sample rate original: {sr_orig}Hz, degradado: {sr_deg}Hz")
            print(f"    RMS original: {rms_orig_n:.8f}  RMS degradado: {rms_deg:.8f}")
            print(f"    diferencia relativa: {diff_rel:+.4%}  ({db:+.2f} dB)")
            if abs(diff_rel) < 1e-4:
                print("    ALERTA: bytes distintos pero RMS practicamente identica (<0.01%) -- revisar si el cambio es real/audible")


def evaluar_version(detector: DetectorComportamiento, ruta: Path) -> dict:
    canal_caller, canal_callee, sr = cargar_canales(str(ruta))
    senal = detector.analizar(canal_caller, canal_callee, sr)
    resultado = fusionar(senal.score)
    return {"score": senal.score, "confidence": resultado.confidence, "is_synthetic": resultado.is_synthetic}


def procesar_llamada(
    detector: DetectorComportamiento, fila: dict, audio_dir: Path, audio_salida_dir: Path
) -> dict | None:
    anon_id = fila["anon_id"]
    ruta_original = audio_dir / f"{anon_id}.wav"
    if not ruta_original.exists():
        print(f"  {anon_id}: audio no encontrado en {ruta_original}, se omite")
        return None

    resultados = {}
    try:
        resultados["original"] = evaluar_version(detector, ruta_original)
        for nombre in NOMBRES_DEGRADACIONES:
            # Se guarda en audio_salida_dir (no en un directorio temporal) para
            # poder reusar estos WAVs como demo en el frontend -- no cambia
            # como se generan ni se evaluan, solo donde quedan.
            ruta_degradada = audio_salida_dir / f"{anon_id}_{nombre}.wav"
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
    parser.add_argument("--audio-salida-dir", default=str(AUDIO_SALIDA_DIR_POR_DEFECTO))
    args = parser.parse_args()

    audio_salida_dir = Path(args.audio_salida_dir)
    audio_salida_dir.mkdir(parents=True, exist_ok=True)

    filas_val = leer_manifest(Path(args.manifest), split="val")
    print(f"Llamadas en split=val: {len(filas_val)}")

    muestra = muestrear_balanceado(filas_val, args.muestra, args.semilla)
    print(f"Muestra balanceada: {len(muestra)} llamadas ({args.muestra // 2} human / {args.muestra // 2} synthetic)")
    print(f"WAVs degradados se guardan en: {audio_salida_dir}")

    if muestra:
        primera = muestra[0]
        ruta_prueba = Path(args.audio_dir) / f"{primera['anon_id']}.wav"
        if ruta_prueba.exists():
            verificar_sanity(ruta_prueba, primera["anon_id"])
        else:
            print(f"Sanity check omitido: audio no encontrado en {ruta_prueba}")

    detector = DetectorComportamiento()
    registros = []
    for fila in muestra:
        registro = procesar_llamada(detector, fila, Path(args.audio_dir), audio_salida_dir)
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
