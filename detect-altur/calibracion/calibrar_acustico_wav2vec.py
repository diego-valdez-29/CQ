"""calibrar_acustico_wav2vec.py — corre SOLO DetectorAcusticoWav2Vec (modelo
preentrenado MelodyMachine/Deepfake-audio-detection-V2) sobre las mismas 30
llamadas que calibrar_acustico_crudo.py (--muestra 30, --semilla 0), para
comparar su separacion human/synthetic con la del detector MFCC actual antes
de decidir si lo reemplaza. No repite Whisper ni Spark: usa la misma
seleccion balanceada sobre el manifest y solo corre DetectorAcusticoWav2Vec.

Los primeros N segundos de RELOJ de canal_caller resultaron ser mayormente
silencio (el caller escucha al agente antes de responder — ver diagnostico
de VAD previo), asi que en vez de eso usa el VAD ya existente en
DetectorComportamiento (_timestamps_habla) para extraer y CONCATENAR los
segmentos de HABLA REAL de canal_caller hasta acumular al menos
MINIMO_HABLA_S segundos de voz (no silencio). Si una llamada no llega a ese
minimo en toda su duracion, usa lo que haya y lo marca en el detalle.

No se ejecuta automaticamente. Correr manualmente con:
    python calibrar_acustico_wav2vec.py
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# Este script vive en calibracion/, pero "app" esta un nivel arriba (raiz del
# proyecto). Python solo agrega el directorio del propio script a sys.path,
# no el cwd, asi que sin esto el import de abajo falla al correrlo desde
# fuera de calibracion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.acustico_wav2vec import DetectorAcusticoWav2Vec
from app.deteccion.comportamiento import DetectorComportamiento
from app.main import cargar_canales

MANIFEST = Path("/home/andres/hackmty26/manifest.csv")
AUDIO_DIR = Path("/home/andres/hackmty26/audio")
SALIDA = Path(__file__).resolve().parent / "calibracion_acustico_wav2vec.json"

MUESTRA = 30
SEMILLA = 0
MINIMO_HABLA_S = 5.0  # segundos de habla real (no silencio) a acumular por llamada


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


def extraer_habla_concatenada(
    canal_caller: np.ndarray, sr: int, vad: DetectorComportamiento, minimo_s: float
) -> dict:
    """Usa el VAD de DetectorComportamiento para ubicar los segmentos de
    habla real de canal_caller y los concatena en orden hasta acumular al
    menos minimo_s segundos. Si la llamada no tiene suficiente habla, regresa
    lo que haya (puede ser un array vacio si no se detecto habla alguna).
    """
    audio_16k = vad._resamplear(canal_caller, sr)
    sr_vad = DetectorComportamiento.SR_VAD
    habla = vad._timestamps_habla(audio_16k)

    trozos = []
    duracion_acumulada = 0.0
    for seg in habla:
        inicio_muestra = int(seg["start"] * sr_vad)
        fin_muestra = int(seg["end"] * sr_vad)
        trozo = audio_16k[inicio_muestra:fin_muestra]
        trozos.append(trozo)
        duracion_acumulada += len(trozo) / sr_vad
        if duracion_acumulada >= minimo_s:
            break

    audio_concatenado = np.concatenate(trozos) if trozos else np.array([], dtype=np.float32)
    return {
        "audio": audio_concatenado,
        "sr": sr_vad,
        "duracion_habla_usada_s": duracion_acumulada,
        "numero_segmentos_totales_llamada": len(habla),
        "numero_segmentos_usados": len(trozos),
        "alcanzo_minimo": duracion_acumulada >= minimo_s,
    }


def reportar_separacion(resultados: list[dict]) -> None:
    """Mismo formato que reportar_separacion_por_senal() en calibrar.py,
    para comparar lado a lado con la señal 'acustico' (MFCC) actual.
    """
    procesados = [r for r in resultados if "score" in r]
    y = np.array([1 if r["label"] == "synthetic" else 0 for r in procesados])
    scores = np.array([r["score"] for r in procesados])
    h, s = scores[y == 0], scores[y == 1]
    print(f"\n=== separacion human/synthetic: acustico_wav2vec (>= {MINIMO_HABLA_S:.0f}s de habla real, VAD) ===")
    print(
        f"acustico_wav2vec: humano mean={h.mean():.3f} std={h.std():.3f} | "
        f"sintetico mean={s.mean():.3f} std={s.std():.3f} | "
        f"|diff medias|={abs(s.mean() - h.mean()):.3f}"
    )


def main() -> None:
    filas_train = leer_manifest(MANIFEST, split="train")
    muestra = muestrear_balanceado(filas_train, MUESTRA, SEMILLA)
    print(f"Llamadas en split=train: {len(filas_train)} — muestra: {len(muestra)}")

    vad = DetectorComportamiento()
    detector = DetectorAcusticoWav2Vec()
    resultados = []
    t_inicio = time.time()

    for fila in muestra:
        anon_id = fila["anon_id"]
        ruta = AUDIO_DIR / f"{anon_id}.wav"
        try:
            canal_caller, canal_callee, sr = cargar_canales(str(ruta))
            extraccion = extraer_habla_concatenada(canal_caller, sr, vad, MINIMO_HABLA_S)
            senal = detector.analizar(extraccion["audio"], canal_callee, extraccion["sr"])
        except Exception as exc:
            resultados.append({
                "anon_id": anon_id,
                "label": fila["label"],
                "error": str(exc),
            })
            print(f"{anon_id} ({fila['label']}): ERROR {exc!r}")
            continue

        if "error" in senal.detalle:
            resultados.append({
                "anon_id": anon_id,
                "label": fila["label"],
                "error": senal.detalle["error"],
                "duracion_habla_usada_s": extraccion["duracion_habla_usada_s"],
                "alcanzo_minimo": extraccion["alcanzo_minimo"],
            })
            print(
                f"{anon_id} ({fila['label']}): ERROR {senal.detalle['error']} "
                f"(duracion_habla_usada_s={extraccion['duracion_habla_usada_s']:.2f})"
            )
            continue

        resultados.append({
            "anon_id": anon_id,
            "label": fila["label"],
            "score": senal.score,
            "duracion_habla_usada_s": extraccion["duracion_habla_usada_s"],
            "numero_segmentos_totales_llamada": extraccion["numero_segmentos_totales_llamada"],
            "numero_segmentos_usados": extraccion["numero_segmentos_usados"],
            "alcanzo_minimo": extraccion["alcanzo_minimo"],
            "salida_cruda": senal.detalle.get("salida_cruda"),
        })
        aviso = "" if extraccion["alcanzo_minimo"] else " [NO ALCANZO EL MINIMO DE HABLA]"
        print(
            f"{anon_id} ({fila['label']}): score={senal.score:.4f} "
            f"duracion_habla_usada_s={extraccion['duracion_habla_usada_s']:.2f}{aviso} "
            f"salida_cruda={senal.detalle.get('salida_cruda')}"
        )

    tiempo_total_s = time.time() - t_inicio

    SALIDA.write_text(json.dumps(resultados, indent=2))
    print(f"\nGuardado en {SALIDA} ({len(resultados)} llamadas)")
    print(f"Ventana usada: >= {MINIMO_HABLA_S:.0f}s de habla real de canal_caller (VAD, concatenada)")
    print(f"Tiempo total: {tiempo_total_s:.1f}s ({tiempo_total_s / len(muestra):.1f}s/llamada en promedio)")

    procesados = [r for r in resultados if "score" in r]
    con_error = [r for r in resultados if "error" in r]
    if con_error:
        print(f"({len(con_error)} llamadas fallaron, ver 'error' en {SALIDA})")

    no_alcanzaron = [r for r in procesados if not r.get("alcanzo_minimo", True)]
    if no_alcanzaron:
        print(
            f"({len(no_alcanzaron)}/{len(procesados)} llamadas no llegaron a "
            f"{MINIMO_HABLA_S:.0f}s de habla real y se usaron con menos)"
        )

    if len(procesados) < 2:
        print("Muy pocos resultados utilizables para reportar separacion.")
        return

    reportar_separacion(resultados)


if __name__ == "__main__":
    main()
