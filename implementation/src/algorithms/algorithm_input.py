from dataclasses import dataclass, field
from typing import List

from cost_functions.cost_function import CostFunction, GaussianCostFunction


@dataclass
class AlgorithmInput:
    signal: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)
    cost_function: CostFunction = field(default_factory=GaussianCostFunction, compare=False, hash=False, repr=False)
    penalization: float = 1.0
    max_amount_changepoints: int = 50

    def initialize(self):
        self.cost_function.precompute(self.signal)





