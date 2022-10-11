import math
from dataclasses import dataclass
from typing import Tuple

from cases.case import Case
from cost_functions.cost_function import KernelBasedCostFunction
from solution.algorithm_input import AlgorithmInput
from solution.binary_segmentation import BinarySegmentation
from solution.solution import Solution
from solution.solver import Solver
from solution.suboptimal_partition_changepoints_in_state_divide_and_conquer_optimzation import DynamicProgrammingDivideAndConquer
from utils.constants import Constants
from visualization.visualization_script import visualize_elbow


@dataclass
class PenalizationSelector:

    def select_penalization(self, case: Case) -> Tuple[float, int]:
        """
        Selects reasonable penalization values and maximum amount of
        changepoints for a given case.
        :param case: case to solve
        :return: a pair with the penalization and max amount of changepoints.
        """


class ElbowPenalizationSelector(PenalizationSelector):
    """
    It uses the solutions that run relatively quickly, even if they are suboptimal, in
    order to choose suitable values of total amount of changepoints and penalization that can
    be used as input to other algorithms.

    Once it has the objective values of the solutions obtained for each amount of changepoints, it
    selects one that seems suitable in terms of the 'elbow method': https://en.wikipedia.org/wiki/Elbow_method_(clustering)
    """
    threshold: float = 1.01

    def select_penalization(self, case: Case, visualize: bool = False) -> Tuple[float, int]:
        """
        It runs dynamic programming suboptimal algorithm to find the solution for each amount of changepoint,
        and the uses binary search to find a suitable penalization for a binary segmentation algorithm to achieve
        such desired amount of changepoitns previously found with the elbow method.
        :param case: input case.
        :param visualize: boolean deciding whether the 'elbow figure' should be visualized or not.
        :return: A tuple with a pair of suitable values for penalization and amount of changepoints.
        """
        changepoints_bound: int = min(Constants.changepoints_bound, math.floor(math.sqrt(case.size)))
        default_cost_function: KernelBasedCostFunction = KernelBasedCostFunction()
        algorithm_input = AlgorithmInput(case=case, cost_function=default_cost_function, max_amount_changepoints=changepoints_bound)
        algorithm_input.initialize()
        greedy_solver_dynamic_programming: Solver = DynamicProgrammingDivideAndConquer(algorithm_input=algorithm_input)
        solution_dynamic_programming: Solution = greedy_solver_dynamic_programming.solve()
        amount_changepoints = len(solution_dynamic_programming.changepoints)
        objective_values = [solution_dynamic_programming.metrics.best_prefix[k][case.size] for k in range(amount_changepoints)]
        guessed_checkpoints = amount_changepoints
        for k in range(1, amount_changepoints - 1):
            relatively_linear = float(objective_values[k - 1] - objective_values[k]) < float(objective_values[k] - objective_values[k + 1]) * self.threshold
            decreasing_substantially = objective_values[k - 1] - objective_values[k] > objective_values[0] * (self.threshold - 1.0)
            if relatively_linear and not decreasing_substantially:
                guessed_checkpoints = k - 1
                break
        if visualize:
            visualize_elbow(case, solution_dynamic_programming, guessed_checkpoints)
        greedy_solver_binary_segmentation: Solver = BinarySegmentation(algorithm_input=algorithm_input)
        lower_penalization, upper_penalization = 0.0, float(case.size) * algorithm_input.cost_function.range_cost(0, case.size)
        while upper_penalization - lower_penalization > Constants.epsilon:
            penalization = (lower_penalization + upper_penalization) / 2.0
            algorithm_input.penalization = penalization
            solution_binary_segmentation: Solution = greedy_solver_binary_segmentation.solve()
            if len(solution_binary_segmentation.changepoints) > guessed_checkpoints:
                lower_penalization = penalization
            else:
                upper_penalization = penalization

        return (lower_penalization + upper_penalization) / 2.0, guessed_checkpoints
