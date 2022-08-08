from dataclasses import dataclass
from typing import List

from metrics.metrics import Metrics
from solution.solution import Solution
from solution.solver import Solver


def retrieve_checkpoints(attained_best: List[int]) -> List[int]:
    """
    It calculated the changepoints by following the attained best
    from the solution to the whole signal.
    :param attained_best: array that holds for the ith prefix the position of the last changepoint.
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

    def solve(self) -> Solution:
        self.algorithm_input.initialize()
        length = self.algorithm_input.case.size + 1  # from [0, 0) to [0,n), note that the last position is at index n-1.
        best_prefix = [0 for _ in range(length)]
        attained_best = [-1 for _ in range(length)]
        for end in range(1, length):
            best_prefix[end], attained_best[end] = min(
                [(best_prefix[i] + self.algorithm_input.cost_function.range_cost(i, end) + self.algorithm_input.penalization, i) for i in range(end)])
        print(retrieve_checkpoints(attained_best))
        return Solution(retrieve_checkpoints(attained_best), Metrics(best_prefix[length - 1], self.name))
