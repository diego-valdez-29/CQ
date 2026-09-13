from app.dominio.modelos import ResultadoDeteccion, SenalScore


class Fusion:
    def __init__(self, pesos: dict[str, float], umbral: float):
        self.pesos = pesos
        self.umbral = umbral

    def combinar(self, senales: list[SenalScore]) -> ResultadoDeteccion:
        suma_pesos = sum(self.pesos.get(s.nombre, 0.0) for s in senales)
        if suma_pesos == 0:
            confidence = sum(s.score for s in senales) / len(senales) if senales else 0.0
        else:
            confidence = sum(
                s.score * self.pesos.get(s.nombre, 0.0) for s in senales
            ) / suma_pesos

        return ResultadoDeteccion(
            is_synthetic=confidence >= self.umbral,
            confidence=confidence,
            senales=senales,
        )
