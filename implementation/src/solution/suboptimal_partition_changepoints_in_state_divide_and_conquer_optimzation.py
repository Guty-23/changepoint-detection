from dataclasses import dataclass

from metrics.metrics import Metrics
from solution.optimal_partition_changepoints_in_state import DynamicProgrammingChangepointsInState
from solution.solution import Solution


@dataclass
class DynamicProgrammingDivideAndConquer(DynamicProgrammingChangepointsInState):
    """ Implementation of Dynamic programming approach, it has
    an O(Dn^2) worst case time complexity, where D is a bound to the amount of changepoints.
    In the worst case in which D = O(n), we have O(n^3) complexity."""

    name: str = 'suboptimal_partition_divide_and_conquer'

    def calculate_range(self, changepoints: int, begin_endpoint: int, finish_endpoint: int, begin_search: int, finish_search: int) -> None:
        middle_endpoint = (begin_endpoint + finish_endpoint) // 2
        self.best_prefix[changepoints][middle_endpoint], self.attained_best[changepoints][middle_endpoint] = \
            min([(self.best_prefix[changepoints - 1][i] + self.cost(i, middle_endpoint) + self.algorithm_input.penalization, i)
                 for i in range(begin_search, min(middle_endpoint + 1, finish_search))])
        if middle_endpoint > begin_endpoint:
            self.calculate_range(changepoints, begin_endpoint, middle_endpoint, begin_search, self.attained_best[changepoints][middle_endpoint] + 1)
        if middle_endpoint + 1 < finish_endpoint:
            self.calculate_range(changepoints, middle_endpoint + 1, finish_endpoint, self.attained_best[changepoints][middle_endpoint], finish_search)

    def solve(self) -> Solution:
        self.initialize()
        amount_changepoints = self.algorithm_input.max_amount_changepoints
        for changepoint_used in range(1, amount_changepoints + 1):
            self.calculate_range(changepoint_used, 0, self.length, 0, self.length)
        return Solution(self.retrieve_changepoints(amount_changepoints),
                        Metrics(self.best_prefix[amount_changepoints][self.length - 1], self.name, self.best_prefix))
