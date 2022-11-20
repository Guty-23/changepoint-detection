import math
from dataclasses import dataclass

from metrics.metrics import Metrics
from solution.solution import Solution
from solution.optimal_partition_changepoints_in_state import DynamicProgrammingChangepointsInState


@dataclass
class DynamicProgrammingChangepointsInStatePruned(DynamicProgrammingChangepointsInState):
    """ Implementation of Dynamic programming approach, it has
    an O(Dn^2) worst case time complexity, where D is a bound to the amount of changepoints.
    In the worst case in which K = O(n), we have O(n^3) complexity. Although it checks fewer
    candidates than the pure approach."""

    name: str = 'optimal_partition_changepoints_in_state_pruned'
    k_term: float = 0.0

    def initialize(self) -> None:
        super(DynamicProgrammingChangepointsInStatePruned, self).initialize()
        self.k_term = - 0.01 * math.log(self.length)

    def solve(self) -> Solution:
        self.initialize()
        amount_changepoints = self.algorithm_input.max_amount_changepoints
        for changepoints_used in range(1, amount_changepoints + 1):
            candidates = {0}
            for end in range(1, self.length):
                self.best_prefix[changepoints_used][end], self.attained_best[changepoints_used][end] = min(
                    [(self.best_prefix[changepoints_used - 1][i] + self.cost(i, end) + self.algorithm_input.penalization, i) for i in candidates])
                candidates = {i for i in candidates if
                              self.best_prefix[changepoints_used - 1][i] + self.cost(i, end) + self.k_term <= self.best_prefix[changepoints_used][end]}
                candidates.add(end)
        return Solution(self.retrieve_changepoints(amount_changepoints),
                        Metrics(self.best_prefix[amount_changepoints][self.length - 1], self.name, self.best_prefix))
