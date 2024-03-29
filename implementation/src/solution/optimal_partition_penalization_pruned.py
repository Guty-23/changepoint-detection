import math
import time
from dataclasses import dataclass

from metrics.metrics import Metrics
from solution.solution import Solution
from solution.optimal_partition_penalization import DynamicProgrammingPenalization


@dataclass
class DynamicProgrammingPenalizationPruned(DynamicProgrammingPenalization):
    """ Implementation of Dynamic programming approach, it has
    an O(n^2) worst case time complexity, although it checks fewer
    candidates than the pure approach."""

    name: str = 'optimal_partition_penalization_pruned'
    k_term: float = 0.0

    def initialize(self) -> set[int]:
        super(DynamicProgrammingPenalizationPruned, self).initialize()
        if 'gaussian' in self.algorithm_input.cost_function.name:
            self.k_term = - math.log(self.length)
        return {0}

    def solve(self) -> Solution:
        start_time = time.perf_counter()
        candidates = self.initialize()
        for end in range(1, self.length):
            self.best_prefix[end], self.attained_best[end] = min(
                [(self.best_prefix[i] + self.cost(i, end) + (self.algorithm_input.penalization if i > 0 else 0.0), i) for i in candidates])
            candidates = {i for i in candidates if self.best_prefix[i] + self.cost(i, end) + self.k_term <= self.best_prefix[end]}
            candidates.add(end)
        end_time = time.perf_counter()
        return Solution(self.retrieve_changepoints(), Metrics(self.best_prefix[self.length - 1], self.name, end_time - start_time, []))
