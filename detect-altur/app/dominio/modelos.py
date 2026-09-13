from dataclasses import dataclass, field


@dataclass
class SenalScore:
    nombre: str
    score: float  # 0..1, donde 0 = humano, 1 = sintetico
    detalle: dict = field(default_factory=dict)


@dataclass
class ResultadoDeteccion:
    is_synthetic: bool
    confidence: float
    senales: list[SenalScore]
