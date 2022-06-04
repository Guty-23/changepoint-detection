from dataclasses import dataclass
from typing import List, Tuple

from solution.algorithm_input import AlgorithmInput


@dataclass
class Solver:
    """ Implements a solution to the problem
    of finding changepoints on a given signal."""
    name: str
    algorithm_input: AlgorithmInput

    def set_input(self, algorithm_input: AlgorithmInput) -> None:
        self.algorithm_input = algorithm_input

    def solve(self) -> Tuple[List[int], float]:
        """
        Finds changepoints to the given input.
        :return: A tuple with a list with the indices in the signal where the changepoints are predicted, and the associated cost.
        """

    def cost(self, start: int, end: int) -> float:
        return self.algorithm_input.cost_function.range_cost(start, end)
