from typing import Protocol

import numpy as np

from app.dominio.modelos import SenalScore


class DetectorDeSenal(Protocol):
    def analizar(
        self,
        canal_caller: np.ndarray,
        canal_callee: np.ndarray,
        sr: int,
    ) -> SenalScore: ...
