from dataclasses import dataclass


@dataclass
class Metrics:
    cost: float
    solver_used: str
