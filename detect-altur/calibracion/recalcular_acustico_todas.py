"""recalcular_acustico_todas.py — recalcula SOLO la señal 'acustico' para las
30 llamadas de calibracion_resultados.jsonl con el DetectorAcustico ya
arreglado (z-score en vez de umbral fijo que saturaba en 1.0).

El jsonl actual mezcla 26 filas calculadas con la formula vieja de acustico
y 4 con la formula nueva (la corrida de calibrar.py se reanudo a mitad de
camino usando su modo --reiniciar-si-no-existe). comportamiento y semantico
en esas 30 filas siguen siendo validos (no dependen de acustico), asi que no
hace falta recalcularlos ni volver a llamar a Whisper/Spark — nada mas se
tocan los valores de 'acustico' en scores_full y en cada scores_checkpoint.

Guarda calibracion_resultados_v2.jsonl y corre los pasos 3-6 de calibrar.py
(separacion por señal, pesos calibrados, comparacion, decision temprana)
sobre el resultado.

No se ejecuta automaticamente. Correr manualmente con:
    python recalcular_acustico_todas.py
"""
import json
import sys
from pathlib import Path

# Este script vive en calibracion/, pero "app" esta un nivel arriba (raiz del
# proyecto). Python solo agrega el directorio del propio script a sys.path,
# no el cwd, asi que sin esto el import de abajo falla al correrlo desde
# fuera de calibracion/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deteccion.acustico import DetectorAcustico
from app.main import cargar_canales
from calibrar import (
    calibrar_pesos,
    reportar_comparacion,
    reportar_decision_temprana,
    reportar_separacion_por_senal,
    truncar,
)

AUDIO_DIR = Path("/home/andres/Downloads/audio")
# NOTA: calibracion_resultados.jsonl (el original, contaminado) ya no existe
# en el repo -- este script cumplio su proposito una sola vez para producir
# calibracion_resultados_v2.jsonl y queda aqui solo como registro de como se
# genero. No es re-ejecutable tal cual sin ese archivo de entrada.
ENTRADA = Path(__file__).resolve().parent / "calibracion_resultados.jsonl"
SALIDA = Path(__file__).resolve().parent / "calibracion_resultados_v2.jsonl"


def main() -> None:
    registros = [json.loads(linea) for linea in ENTRADA.read_text().splitlines()]
    print(f"Leidas {len(registros)} filas de {ENTRADA}")

    detector = DetectorAcustico()

    for i, r in enumerate(registros, start=1):
        anon_id = r["anon_id"]
        ruta_audio = AUDIO_DIR / f"{anon_id}.wav"
        canal_caller, canal_callee, sr = cargar_canales(str(ruta_audio))

        acustico_antes = r["scores_full"]["acustico"]
        r["scores_full"]["acustico"] = detector.analizar(canal_caller, canal_callee, sr).score

        for clave, scores_cp in r["scores_checkpoint"].items():
            checkpoint_s = float(clave.rstrip("s"))
            c_caller, c_callee = truncar(canal_caller, canal_callee, sr, checkpoint_s)
            scores_cp["acustico"] = detector.analizar(c_caller, c_callee, sr).score

        print(
            f"  [{i}/{len(registros)}] {anon_id} ({r['label']}): "
            f"acustico {acustico_antes:.4f} -> {r['scores_full']['acustico']:.4f}"
        )

    with open(SALIDA, "w") as f:
        for r in registros:
            f.write(json.dumps(r) + "\n")
    print(f"\nGuardado en {SALIDA}")

    reportar_separacion_por_senal(registros)
    pesos_calibrados, umbral_calibrado = calibrar_pesos(registros)
    reportar_comparacion(registros, pesos_calibrados, umbral_calibrado)
    reportar_decision_temprana(registros, pesos_calibrados)


if __name__ == "__main__":
    main()
