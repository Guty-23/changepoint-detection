from dataclasses import dataclass, field
from typing import List

from metrics.metrics import Metrics
from solution.solution import Solution
from solution.solver import Solver
from utils.constants import Constants


@dataclass
class DynamicProgrammingChangepointsInState(Solver):
    """ Implementation of Dynamic programming approach, it has
    an O(Dn^2) worst case time complexity, where D is a bound to the amount of changepoints.
    In the worst case in which D = O(n), we have O(n^3) complexity.

    Notice that we add the penalization to the cost function so that the results among the different algorithms
    are comparable, since adding a constant to the objective function does not change the decisions."""

    name: str = 'optimal_partition_changepoints_in_state'
    best_prefix: List[List[float]] = field(default_factory=list, compare=False, hash=False, repr=False)
    attained_best: List[List[int]] = field(default_factory=list, compare=False, hash=False, repr=False)
    length: int = 0

    def retrieve_changepoints(self, changepoints_used: int) -> List[int]:
        """
        It calculates the changepoints by following the attained best
        from the solution to the whole signal.
        :param changepoints_used: amount of changepoints in the signal for optimal solution.
        :return: The list with all the optimal changepoints calculated for the amount specified.
        """
        changepoints = []
        actual = self.length - 1
        for changepoint in range(changepoints_used, 0, -1):
            changepoints.append(self.attained_best[changepoint][actual])
            actual = self.attained_best[changepoint][actual]
        return changepoints

    def initialize(self) -> None:
        self.length = self.algorithm_input.case.size + 1  # from [0, 0) to [0,n), note that the last position is at index n-1.
        self.best_prefix = [[Constants.infinity for end in range(self.length)] for changepoint in range(self.algorithm_input.max_amount_changepoints + 1)]
        self.attained_best = [[-2 for end in range(self.length + 1)] for changepoint in range(self.algorithm_input.max_amount_changepoints + 1)]
        self.best_prefix[0] = [self.cost(0, end) for end in range(self.length)]
        self.attained_best[0] = [-1 for end in range(self.length)]

    def solve(self) -> Solution:
        self.initialize()
        amount_changepoints = self.algorithm_input.max_amount_changepoints
        for changepoints_used in range(1, amount_changepoints + 1):
            for end in range(1, self.length):
                self.best_prefix[changepoints_used][end], self.attained_best[changepoints_used][end] = min(
                    [(self.best_prefix[changepoints_used - 1][i] + self.cost(i, end) + self.algorithm_input.penalization, i) for i in range(end)])
        return Solution(self.retrieve_changepoints(amount_changepoints),
                        Metrics(self.best_prefix[amount_changepoints][self.length - 1], self.name, self.best_prefix))
