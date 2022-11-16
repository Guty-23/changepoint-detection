from dataclasses import dataclass
from typing import List


@dataclass
class Metrics:
    cost: float
    solver_used: str
    best_prefix: List[List[float]]
