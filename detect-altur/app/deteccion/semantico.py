import numpy as np

from app.dominio.modelos import SenalScore


class DetectorSemantico:
    def analizar(
        self,
        canal_caller: np.ndarray,
        canal_callee: np.ndarray,
        sr: int,
    ) -> SenalScore:
        return SenalScore(nombre="semantico", score=0.5, detalle={})
