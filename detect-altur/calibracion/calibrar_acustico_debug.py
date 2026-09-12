"""calibrar_acustico_debug.py — diagnostico puntual de por que DetectorAcustico
satura en score=1.0 en llamadas human reales del dataset (8kHz, telefonia).

Corre analizar_debug() (opt-in, no usado por el pipeline normal) sobre las 5
llamadas donde se detecto el patron de saturacion, e imprime una tabla con los
valores crudos que entran al sigmoid de _score_heuristico: varianza_mfcc y
regularidad_espectral (antes del sigmoid), el exponente resultante, y el
score final — junto a los umbrales actuales para comparar a simple vista.

NO se ejecuta automaticamente. Correr manualmente con:
    python calibrar_acustico_debug.py
"""
import sys
from pathlib import Path

# Este script vive en calibracion/, pero "app" esta un nivel arriba (raiz del
# proyecto). Python solo agrega el directorio del propio script a sys.path,
# no el cwd, asi que sin esto el import de abajo falla al correrlo desde
# fuera de calibracion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.acustico import DetectorAcustico
from app.main import cargar_canales

AUDIO_DIR = Path("/home/andres/Downloads/audio")

LLAMADAS = [
    "call_60ffdefc5f5a",  # human, 211s, score reportado 1.0
    "call_893d552b156a",  # human, 153s, score reportado 1.0
    "call_b41c86402714",  # human, 141s, score reportado 1.0
    "call_36640e327eaf",  # human, 159s, score reportado 1.0
    "call_4ac92327c83b",  # human, 152s, score reportado 1.0
]


def main() -> None:
    detector = DetectorAcustico()
    filas = []

    for anon_id in LLAMADAS:
        ruta = AUDIO_DIR / f"{anon_id}.wav"
        canal_caller, canal_callee, sr = cargar_canales(str(ruta))
        _, debug = detector.analizar_debug(canal_caller, canal_callee, sr)
        filas.append((anon_id, debug))

    encabezado = (
        f"{'anon_id':<20} {'varianza_mfcc':>15} {'z_varianza':>12} "
        f"{'regularidad_esp':>16} {'z_regularidad':>14} "
        f"{'exponente':>14} {'score_final':>12}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for anon_id, d in filas:
        if "error" in d:
            print(f"{anon_id:<20} ERROR: {d['error']}")
            continue
        print(
            f"{anon_id:<20} {d['varianza_mfcc']:>15.4f} {d['z_varianza']:>12.4f} "
            f"{d['regularidad_espectral']:>16.4f} {d['z_regularidad']:>14.4f} "
            f"{d['exponente']:>14.4f} {d['score_final']:>12.4f}"
        )

    print()
    print(f"media_varianza actual:     {detector._media_varianza}  std: {detector._std_varianza}")
    print(f"media_regularidad actual:  {detector._media_regularidad}  std: {detector._std_regularidad}")


if __name__ == "__main__":
    main()
