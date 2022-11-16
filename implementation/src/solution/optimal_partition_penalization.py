from dataclasses import dataclass, field
from typing import List

from metrics.metrics import Metrics
from solution.solution import Solution
from solution.solver import Solver


@dataclass
class DynamicProgrammingPenalization(Solver):
    """ Implementation of Dynamic programming approach, it has
    an O(n^2) worst case time complexity."""

    name: str = 'optimal_partition_penalization'
    best_prefix: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)
    attained_best: List[int] = field(default_factory=list, compare=False, hash=False, repr=False)
    length: int = 0

    def retrieve_changepoints(self) -> List[int]:
        """
        It calculates the changepoints by following the attained best
        from the solution to the whole signal.
        :return: The list with all the changepoints calculated.
        """
        changepoints = []
        actual = self.length - 1
        while self.attained_best[actual] != 0:
            changepoints.append(self.attained_best[actual])
            actual = self.attained_best[actual]
        return changepoints

    def initialize(self) -> None:
        self.length = self.algorithm_input.case.size + 1  # from [0, 0) to [0,n), note that the last position is at index n-1.
        self.best_prefix = [0 for _ in range(self.length)]
        self.attained_best = [-1 for _ in range(self.length)]

    def solve(self) -> Solution:
        self.initialize()
        for end in range(1, self.length):
            self.best_prefix[end], self.attained_best[end] = min(
                [(self.best_prefix[i] + self.cost(i, end) + (self.algorithm_input.penalization if i > 0 else 0.0), i) for i in range(end)])
        return Solution(self.retrieve_changepoints(), Metrics(self.best_prefix[self.length - 1], self.name, []))
