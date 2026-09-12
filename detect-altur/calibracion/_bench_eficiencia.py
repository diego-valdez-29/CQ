
"""Harness temporal: 5 pruebas de eficiencia / robustez / factibilidad / latencia.

No es un test de produccion. Corre: python calibracion/_bench_eficiencia.py
desde la raiz de detect-altur.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Stack real vs mocks
# ---------------------------------------------------------------------------
STACK_REAL = {}
try:
    import faster_whisper  # noqa: F401
    import librosa  # noqa: F401

    STACK_REAL["faster_whisper"] = True
    STACK_REAL["librosa"] = True
except Exception as exc:  # pragma: no cover
    STACK_REAL["faster_whisper"] = False
    STACK_REAL["librosa"] = False
    STACK_REAL["import_error"] = repr(exc)

WHISPER_PESOS = False
try:
    from faster_whisper import WhisperModel  # noqa: F401

    STACK_REAL["WhisperModel_import"] = True
except Exception as exc:
    STACK_REAL["WhisperModel_import"] = False
    STACK_REAL["WhisperModel_error"] = repr(exc)


def _hay_pesos_whisper() -> bool:
    home = Path.home()
    candidatos = [
        home / ".cache" / "huggingface",
        home / ".cache" / "whisper",
        home / "AppData" / "Local" / "huggingface",
    ]
    for base in candidatos:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            name = p.name.lower()
            if "whisper" in name and p.is_file() and p.stat().st_size > 1_000_000:
                return True
    return False


WHISPER_PESOS = _hay_pesos_whisper()
STACK_REAL["whisper_pesos_en_cache"] = WHISPER_PESOS

from app.deteccion.semantico import DetectorSemantico, LLM_TIMEOUT_PARED_S

# Evitar importar app.main (fastapi). Valor leido de main.py: CHECKPOINTS_S = (25.0, 40.0)
CHECKPOINTS_S = (25.0, 40.0)

# Importar solo tope/clasificar de calibrar sin arrastrar sklearn/fastapi si faltan.
try:
    from calibracion.calibrar import (  # type: ignore
        TopePorLlamada,
        asegurar_dentro_de_tope,
        clasificar_semantico,
        TOPE_POR_LLAMADA_S,
    )

    CALIBRAR_IMPORT_OK = True
    CALIBRAR_IMPORT_ERR = None
except Exception as exc:
    CALIBRAR_IMPORT_OK = False
    CALIBRAR_IMPORT_ERR = repr(exc)

    TOPE_POR_LLAMADA_S = 90.0

    class TopePorLlamada(Exception):
        pass

    def asegurar_dentro_de_tope(t0: float) -> None:
        if time.time() - t0 > TOPE_POR_LLAMADA_S:
            raise TopePorLlamada(f"excedio {TOPE_POR_LLAMADA_S:.0f}s")

    def clasificar_semantico(detector, segmentos, hasta_s=None):
        if segmentos is None:
            from app.dominio.modelos import SenalScore

            return SenalScore(
                nombre="semantico",
                score=0.5,
                detalle={"fallback": "timeout_whisper"},
            )
        return detector._clasificar_transcript(detector._texto_hasta(segmentos, hasta_s))


FACTOR_SLEEP_POR_S = 0.02  # 1 s de audio ~ 20 ms de "Whisper" simulado
COSTO_VAD_RESAMPLE_MEDIDO: dict[str, float] = {}


def audio_sintetico(duracion_s: float, sr: int = 16000) -> np.ndarray:
    n = int(duracion_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    return (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def segs_ejemplo() -> list[dict]:
    return [
        {"start": 0.0, "end": 8.0, "text": "hola buenos dias"},
        {"start": 8.0, "end": 20.0, "text": "mi nombre es ana"},
        {"start": 20.0, "end": 32.0, "text": "no se esa clave"},
        {"start": 32.0, "end": 48.0, "text": "el saldo es mil pesos"},
        {"start": 48.0, "end": 70.0, "text": "gracias por esperar"},
    ]


class _Resultado:
    def __init__(self, nombre: str, eje: str):
        self.nombre = nombre
        self.eje = eje
        self.ok = False
        self.ms = 0.0
        self.notas = ""

    def as_row(self) -> tuple:
        return (
            self.nombre,
            self.eje,
            "PASS" if self.ok else "FAIL",
            f"{self.ms:.1f}",
            self.notas,
        )


def _mock_transcribe_costo(audio: np.ndarray) -> list[dict]:
    dur = len(audio) / DetectorSemantico.SR_WHISPER
    time.sleep(max(0.001, dur * FACTOR_SLEEP_POR_S))
    # Segmentos proporcionalmente espaciados para poder recortar.
    return [
        {"start": 0.0, "end": min(dur, 25.0), "text": "hola no se el dato del inicio"},
        {"start": 25.0, "end": min(dur, 40.0), "text": "segunda parte invento un numero 42"},
        {"start": 40.0, "end": dur, "text": "cierre de la llamada"},
    ]


def _medir_preparar_audio(duracion_s: float, sr_in: int = 8000) -> float:
    det = DetectorSemantico()
    wav = audio_sintetico(duracion_s, sr=sr_in)
    t0 = time.perf_counter()
    out = det._preparar_audio(wav, sr_in)
    dt = time.perf_counter() - t0
    assert out.dtype == np.float32
    assert abs(len(out) / det.SR_WHISPER - duracion_s) < 0.05
    return dt * 1000.0


# ---------------------------------------------------------------------------
# P1 — eficiencia Whisper 1x vs 3x
# ---------------------------------------------------------------------------
def prueba_1() -> _Resultado:
    r = _Resultado("P1 Whisper 1x vs 3x", "eficiencia")
    det = DetectorSemantico()
    det._transcribir_sin_timeout = _mock_transcribe_costo  # type: ignore[method-assign]
    det._clasificar_con_llm = lambda transcript: (_ for _ in ()).throw(  # type: ignore
        RuntimeError("LLM no debe llamarse: usar regex")
    )

    sr = 16000
    full = audio_sintetico(60.0, sr)
    c25, c40 = full[: 25 * sr], full[: 40 * sr]

    t0 = time.perf_counter()
    s25 = det.analizar(c25, c25, sr)
    s40 = det.analizar(c40, c40, sr)
    sfull = det.analizar(full, full, sr)
    t3x = time.perf_counter() - t0

    t0 = time.perf_counter()
    segs = det._transcribir(full, sr)
    assert segs is not None
    t1x_txt25 = det._clasificar_transcript(det._texto_hasta(segs, 25.0))
    t1x_txt40 = det._clasificar_transcript(det._texto_hasta(segs, 40.0))
    t1x_full = det._clasificar_transcript(det._texto_hasta(segs))
    t1x = time.perf_counter() - t0

    speedup = t3x / t1x if t1x > 0 else float("inf")
    r.ms = t1x * 1000.0
    r.ok = (
        t1x < t3x
        and speedup >= 1.4
        and s25.score == 0.15
        and t1x_txt25.score == 0.15
        and s40.nombre == "semantico"
        and t1x_full.score in (0.15, 0.5)
        and t1x_txt40.nombre == "semantico"
        and sfull.nombre == "semantico"
    )
    r.notas = (
        f"3x={t3x*1000:.0f}ms 1x={t1x*1000:.0f}ms speedup={speedup:.2f}x "
        f"(mock sleep {FACTOR_SLEEP_POR_S*1000:.0f}ms/s audio; "
        f"Whisper real pesos={'si' if WHISPER_PESOS else 'no'}) "
        f"scores 3x=({s25.score},{s40.score},{sfull.score}) "
        f"1x=({t1x_txt25.score},{t1x_txt40.score},{t1x_full.score})"
    )
    return r


# ---------------------------------------------------------------------------
# P2 — timeout de pared Whisper
# ---------------------------------------------------------------------------
def prueba_2() -> _Resultado:
    r = _Resultado("P2 timeout pared Whisper", "latencia")
    det = DetectorSemantico()

    def _cuelga(audio: np.ndarray) -> list[dict]:
        time.sleep(5.0)
        return [{"start": 0.0, "end": 1.0, "text": "nunca"}]

    det._transcribir_sin_timeout = _cuelga  # type: ignore[method-assign]
    det._timeout_whisper_s = staticmethod(lambda duracion_s: 0.2)  # type: ignore

    wav = audio_sintetico(3.0)
    t0 = time.perf_counter()
    senal = det.analizar(wav, wav, 16000)
    elapsed = time.perf_counter() - t0
    r.ms = elapsed * 1000.0
    r.ok = (
        0.15 <= elapsed <= 0.55
        and senal.score == 0.5
        and (senal.detalle or {}).get("fallback") == "timeout_whisper"
    )
    r.notas = (
        f"elapsed={elapsed*1000:.0f}ms score={senal.score} "
        f"detalle={senal.detalle} (objetivo 200-500ms, hang era 5s)"
    )
    det._executor.shutdown(wait=False)
    return r


# ---------------------------------------------------------------------------
# P3 — fallbacks
# ---------------------------------------------------------------------------
def prueba_3() -> _Resultado:
    r = _Resultado("P3 fallbacks robustez", "robustez")
    det = DetectorSemantico()
    t0 = time.perf_counter()
    fallos = []

    vacio = det._clasificar_transcript("")
    if not (vacio.score == 0.5 and vacio.detalle.get("error") == "transcript_vacio"):
        fallos.append(f"vacio={vacio}")

    honesto = det._clasificar_transcript("la verdad es que no sé esa informacion")
    if not (
        honesto.score == 0.15
        and honesto.detalle.get("metodo") == "regex_honesto"
    ):
        fallos.append(f"regex={honesto}")

    prev = os.environ.pop("SPARK_API_KEY", None)
    try:
        llm = det._clasificar_transcript(
            "el numero de cuenta es 123456 y la fecha de nacimiento 1 de enero"
        )
    finally:
        if prev is not None:
            os.environ["SPARK_API_KEY"] = prev
    if not (llm.score == 0.5 and llm.detalle.get("error") == "llm_no_disponible"):
        fallos.append(f"llm_key={llm}")

    def _llm_lento(transcript: str):
        time.sleep(5.0)
        raise RuntimeError("no deberia verse")

    det._clasificar_con_llm = _llm_lento  # type: ignore[method-assign]
    t_llm0 = time.perf_counter()
    to = det._clasificar_transcript("invento un dato concreto 999")
    t_llm = time.perf_counter() - t_llm0
    if not (
        to.score == 0.5
        and to.detalle.get("fallback") == "timeout_pared"
        and 2.5 <= t_llm <= 3.6
    ):
        fallos.append(f"timeout_llm={to} t={t_llm:.3f}s (pared={LLM_TIMEOUT_PARED_S})")

    t_tope = time.time() - (TOPE_POR_LLAMADA_S + 1.0)
    try:
        asegurar_dentro_de_tope(t_tope)
        fallos.append("tope_no_lanzo")
    except TopePorLlamada:
        pass

    t_ok = time.time()
    try:
        asegurar_dentro_de_tope(t_ok)
    except Exception as exc:
        fallos.append(f"tope_falso_positivo={exc!r}")

    none_segs = clasificar_semantico(det, None)
    if not (
        none_segs.score == 0.5
        and none_segs.detalle.get("fallback") == "timeout_whisper"
    ):
        fallos.append(f"segs_none={none_segs}")

    r.ms = (time.perf_counter() - t0) * 1000.0
    r.ok = not fallos
    r.notas = (
        f"casos: vacio=0.5 regex=0.15 llm_no_disponible=0.5 "
        f"timeout_llm={t_llm*1000:.0f}ms tope={TOPE_POR_LLAMADA_S:.0f}s "
        f"import_calibrar={'ok' if CALIBRAR_IMPORT_OK else CALIBRAR_IMPORT_ERR} "
        f"fallos={fallos or 'ninguno'}"
    )
    det._executor.shutdown(wait=False)
    return r


# ---------------------------------------------------------------------------
# P4 — recorte de segmentos / equivalencia 1x vs 3x texto
# ---------------------------------------------------------------------------
def prueba_4() -> _Resultado:
    r = _Resultado("P4 recorte segmentos 25/40", "factibilidad")
    det = DetectorSemantico()
    segs = segs_ejemplo()
    t0 = time.perf_counter()

    t25 = DetectorSemantico._texto_hasta(segs, 25.0)
    t40 = DetectorSemantico._texto_hasta(segs, 40.0)
    tfull = DetectorSemantico._texto_hasta(segs)

    esperado_25 = "hola buenos dias mi nombre es ana no se esa clave"
    esperado_40 = esperado_25 + " el saldo es mil pesos"
    esperado_full = esperado_40 + " gracias por esperar"

    eq_25 = t25 == esperado_25
    eq_40 = t40 == esperado_40
    eq_full = tfull == esperado_full

    c25 = det._clasificar_transcript(t25)
    c40 = det._clasificar_transcript(t40)
    cfull = det._clasificar_transcript(tfull)

    # Misma transcripcion recortada == tres "transcripciones" independientes
    # (mismo texto => mismo score).
    c25_b = det._clasificar_transcript(esperado_25)
    equiv = (
        c25.score == c25_b.score == 0.15
        and c25.detalle.get("metodo") == "regex_honesto"
        and c40.score == 0.15
        and cfull.score == 0.15
    )

    via_calibrar = clasificar_semantico(det, segs, hasta_s=25.0)
    via_calibrar_ok = via_calibrar.score == 0.15

    r.ms = (time.perf_counter() - t0) * 1000.0
    r.ok = eq_25 and eq_40 and eq_full and equiv and via_calibrar_ok
    r.notas = (
        f"eq_texto 25/40/full={eq_25}/{eq_40}/{eq_full} "
        f"scores=({c25.score},{c40.score},{cfull.score}) "
        f"clasificar_semantico_25={via_calibrar.score} "
        f"texto25={t25!r}"
    )
    return r


# ---------------------------------------------------------------------------
# P5 — costo produccion (3x) vs calibracion (1x)
# ---------------------------------------------------------------------------
def prueba_5() -> _Resultado:
    r = _Resultado("P5 prod 3x vs calib 1x", "latencia/eficiencia")
    det = DetectorSemantico()
    det._transcribir_sin_timeout = _mock_transcribe_costo  # type: ignore[method-assign]

    def costo_whisper_mock(canal: np.ndarray, sr: int) -> float:
        t0 = time.perf_counter()
        segs = det._transcribir(canal, sr)
        return time.perf_counter() - t0, segs

    sr = 16000
    corta = audio_sintetico(10.0, sr)
    larga = audio_sintetico(60.0, sr)

    # Produccion decidir_secuencial: si dur <= checkpoint, lo salta.
    # Corta 10s: solo full (1x). Larga 60s: 25 + 40 + full (3x).
    t0 = time.perf_counter()
    if 10.0 > CHECKPOINTS_S[0]:
        costo_whisper_mock(corta[: int(CHECKPOINTS_S[0] * sr)], sr)
        costo_whisper_mock(corta[: int(CHECKPOINTS_S[1] * sr)], sr)
    t_prod_corta, _ = costo_whisper_mock(corta, sr)
    # arriba solo full; rehacer limpio:
    t0 = time.perf_counter()
    costo_whisper_mock(corta, sr)  # unico checkpoint: fin
    t_prod_corta = time.perf_counter() - t0

    t0 = time.perf_counter()
    for cp in CHECKPOINTS_S:
        costo_whisper_mock(larga[: int(cp * sr)], sr)
    costo_whisper_mock(larga, sr)
    t_prod_larga = time.perf_counter() - t0

    t0 = time.perf_counter()
    costo_whisper_mock(corta, sr)
    t_cal_corta = time.perf_counter() - t0

    t0 = time.perf_counter()
    costo_whisper_mock(larga, sr)
    t_cal_larga = time.perf_counter() - t0

    # Resample real (8 kHz -> 16 kHz), parte VAD/resample medible en hardware.
    try:
        ms_rs_10 = _medir_preparar_audio(10.0, sr_in=8000)
        ms_rs_60 = _medir_preparar_audio(60.0, sr_in=8000)
        COSTO_VAD_RESAMPLE_MEDIDO["resample_10s_ms"] = ms_rs_10
        COSTO_VAD_RESAMPLE_MEDIDO["resample_60s_ms"] = ms_rs_60
    except Exception as exc:
        ms_rs_10 = ms_rs_60 = -1.0
        COSTO_VAD_RESAMPLE_MEDIDO["error"] = repr(exc)

    # Modelo analitico: costo ~ duracion (misma constante).
    modelo_prod_larga = FACTOR_SLEEP_POR_S * (25 + 40 + 60)
    modelo_cal_larga = FACTOR_SLEEP_POR_S * 60
    modelo_ratio = modelo_prod_larga / modelo_cal_larga
    ratio_medido = t_prod_larga / t_cal_larga if t_cal_larga > 0 else float("inf")

    r.ms = t_prod_larga * 1000.0
    r.ok = (
        t_prod_larga > t_cal_larga * 1.5
        and abs(ratio_medido - modelo_ratio) / modelo_ratio < 0.35
        and abs(t_prod_corta - t_cal_corta) / max(t_cal_corta, 1e-6) < 0.5
        and ms_rs_60 > 0
    )
    r.notas = (
        f"larga prod={t_prod_larga*1000:.0f}ms calib={t_cal_larga*1000:.0f}ms "
        f"ratio={ratio_medido:.2f}x (modelo {modelo_ratio:.2f}x = "
        f"{modelo_prod_larga*1000:.0f}/{modelo_cal_larga*1000:.0f}ms) "
        f"corta prod={t_prod_corta*1000:.0f}ms calib={t_cal_corta*1000:.0f}ms "
        f"(10s no alcanza checkpoints {CHECKPOINTS_S} => 1x==1x) "
        f"resample_real 10s={ms_rs_10:.1f}ms 60s={ms_rs_60:.1f}ms "
        f"checkpoints={CHECKPOINTS_S}"
    )
    det._executor.shutdown(wait=False)
    return r


def _tabla(rows: list[_Resultado]) -> str:
    headers = ("prueba", "eje", "resultado", "ms", "notas")
    data = [headers] + [row.as_row() for row in rows]
    widths = [max(len(str(r[i])) for r in data) for i in range(5)]
    # notas puede ser enorme; cap display width for col 4 in header only
    widths[4] = min(widths[4], 90)
    lines = []
    for i, row in enumerate(data):
        cells = []
        for j, cell in enumerate(row):
            s = str(cell)
            if j == 4 and len(s) > 90:
                s = s[:87] + "..."
            cells.append(s.ljust(widths[j]))
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> int:
    print("=== detect-altur bench eficiencia/robustez/factibilidad/latencia ===")
    print(f"stack_real={STACK_REAL}")
    print(f"import_calibrar={CALIBRAR_IMPORT_OK} err={CALIBRAR_IMPORT_ERR}")
    print()

    resultados: list[_Resultado] = []
    for fn in (prueba_1, prueba_2, prueba_3, prueba_4, prueba_5):
        try:
            resultados.append(fn())
        except Exception as exc:
            bad = _Resultado(fn.__name__, "error")
            bad.ok = False
            bad.notas = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            resultados.append(bad)
        row = resultados[-1]
        print(f"[{'PASS' if row.ok else 'FAIL'}] {row.nombre}  {row.ms:.1f}ms")
        print(f"       {row.notas}")
        print()

    print(_tabla(resultados))
    n_ok = sum(1 for x in resultados if x.ok)
    print(f"\n{n_ok}/{len(resultados)} PASS")
    return 0 if n_ok == len(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
