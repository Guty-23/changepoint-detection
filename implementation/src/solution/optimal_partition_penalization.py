from dataclasses import dataclass, field
from typing import List

from metrics.metrics import Metrics
from solution.solution import Solution
from solution.solver import Solver


def retrieve_checkpoints(attained_best: List[int]) -> List[int]:
    """
    It calculates the changepoints by following the attained best
    from the solution to the whole signal.
    :param attained_best: array that holds for the i-th prefix the position of the last changepoint.
    :return: The list with all the changepoints calculated.
    """
    changepoints = []
    actual = len(attained_best) - 1
    while attained_best[actual] != 0:
        changepoints.append(attained_best[actual])
        actual = attained_best[actual]
    return changepoints


@dataclass
class OptimalPartition(Solver):
    """ Implementation of Dynamic programming approach, it has
    an O(n^2) worst case time complexity."""

    name: str = 'optimal_partition_penalization'
    best_prefix: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)
    attained_best: List[int] = field(default_factory=list, compare=False, hash=False, repr=False)
    length: int = 0

    def initialize(self):
        self.length = self.algorithm_input.case.size + 1  # from [0, 0) to [0,n), note that the last position is at index n-1.
        self.best_prefix = [0 for _ in range(self.length)]
        self.attained_best = [-1 for _ in range(self.length)]

    def solve(self) -> Solution:
        self.initialize()
        for end in range(1, self.length):
            self.best_prefix[end], self.attained_best[end] = min(
                [(self.best_prefix[i] + self.cost(i, end) + self.algorithm_input.penalization, i) for i in range(end)])
        return Solution(retrieve_checkpoints(self.attained_best), Metrics(self.best_prefix[self.length - 1], self.name))
