import math
from dataclasses import dataclass, field
from typing import List

from metrics.metrics import Metrics
from solution.algorithm_input import AlgorithmInput
from solution.solution import Solution
from solution.solver import Solver
from solution.optimal_partition_penalization import retrieve_checkpoints
from utils.constants import Constants


@dataclass
class OptimalPartitionPruned(Solver):
    """ Implementation of Dynamic programming approach, it has
    an O(n^2) worst case time complexity, although it checks fewer
    candidates than the pure approach."""

    name: str = 'optimal_partition_penalization_pruned'
    best_prefix: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)
    attained_best: List[int] = field(default_factory=list, compare=False, hash=False, repr=False)
    length: int = 0
    k_term: float = 0.0

    def initialize(self) -> set[int]:
        self.length = self.algorithm_input.case.size + 1  # from [0, 0) to [0,n), note that the last position is at index n-1.
        self.best_prefix = [0.0 for _ in range(self.length)]
        self.attained_best = [-1 for _ in range(self.length)]
        if 'kernel' not in self.algorithm_input.cost_function.name:
            self.k_term = - math.log(self.length)
        return {0}

    def solve(self) -> Solution:
        candidates = self.initialize()
        for end in range(1, self.length):
            self.best_prefix[end], self.attained_best[end] = min(
                [(self.best_prefix[i] + self.cost(i, end) + self.algorithm_input.penalization, i) for i in candidates])
            candidates = {i for i in candidates if self.best_prefix[i] + self.cost(i, end) + self.k_term <= self.best_prefix[end]}
            candidates.add(end)
        return Solution(retrieve_checkpoints(self.attained_best), Metrics(self.best_prefix[self.length - 1], self.name))
