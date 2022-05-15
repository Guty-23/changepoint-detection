from dataclasses import dataclass
from typing import Tuple, List

from solution.solver import Solver


@dataclass
class BinarySegmentation(Solver):
    """ Implementation of greedy binary segmentation Algorithm,
    expected O(n log(n)) runtime, although it has an O(n^2) worst
    case time complexity."""

    def split_cost(self, start: int, split_position: int, end: int) -> float:
        return self.cost(start, split_position) + \
               self.cost(split_position + 1, end) + \
               self.algorithm_input.penalization

    def solve_range(self, start: int, end: int, total_cost: float, changepoints: List[int]) -> Tuple[List[int], float]:
        candidate_cost, candidate = min([(self.split_cost(start, position, end), position) for position in range(start + 1, end)])
        if candidate_cost < self.cost(start, end):
            changepoints_left, total_cost_left = self.solve_range(start, candidate, total_cost, changepoints)
            changepoints_right, total_cost_right = self.solve_range(candidate + 1, end, total_cost, changepoints)
            changepoints.append(candidate)
            total_cost += total_cost_left + total_cost_right + self.algorithm_input.penalization
        else:
            total_cost += self.cost(start, end)
        return changepoints, total_cost

    def solve(self) -> Tuple[List[int], float]:
        self.algorithm_input.initialize()
        return self.solve_range(0, len(self.algorithm_input.case.signal), 0.0, [])