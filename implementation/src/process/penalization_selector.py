import math
from dataclasses import dataclass
from typing import Tuple, List, Callable

from cases.case import Case
from cost_functions.cost_function import KernelBasedCostFunction
from cost_functions.kernels import LaplaceKernel
from solution.algorithm_input import AlgorithmInput
from solution.binary_segmentation import BinarySegmentation
from solution.optimal_partition_changepoints_in_state import DynamicProgrammingChangepointsInState
from solution.solution import Solution
from solution.solver import Solver
from solution.suboptimal_partition_changepoints_in_state_divide_and_conquer_optimzation import DynamicProgrammingDivideAndConquer
from utils.constants import Constants
from visualization.visualization_script import visualize_elbow, visualize_silhouette


def obtain_solution_properties(case: Case, account_penalization: bool) -> Tuple[List[float], List[List[int]], Solution, AlgorithmInput]:
    """
    It obtains the solution for different amount of changepoints.
    :param case: case to solve.
    :param account_penalization: boolean deciding whether penalization for each changepoint should count in the objective
    :return: The list of objective values, and changepoints obtained for each specified amount and extra values for simplicity of code.
    """
    changepoints_bound: int = min(Constants.changepoints_bound, math.floor(math.sqrt(case.size)))
    default_cost_function: KernelBasedCostFunction = KernelBasedCostFunction()
    algorithm_input = AlgorithmInput(case=case, cost_function=default_cost_function, max_amount_changepoints=changepoints_bound)
    algorithm_input.initialize()
    greedy_solver_dynamic_programming: DynamicProgrammingChangepointsInState = DynamicProgrammingDivideAndConquer(algorithm_input=algorithm_input)
    solution_dynamic_programming: Solution = greedy_solver_dynamic_programming.solve()
    amount_changepoints = len(solution_dynamic_programming.changepoints)
    penalization = algorithm_input.penalization if not account_penalization else 0.0
    objective_values = [solution_dynamic_programming.metrics.best_prefix[k][case.size] - k * penalization for k in range(amount_changepoints)]
    # print(objective_values)
    changepoints_values = [greedy_solver_dynamic_programming.retrieve_changepoints(k) for k in range(amount_changepoints)]
    return objective_values, changepoints_values, solution_dynamic_programming, algorithm_input


def obtain_penalization_from_changepoints(case: Case, algorithm_input: AlgorithmInput, guessed_changepoints: int) -> float:
    lower_penalization, upper_penalization = 0.0, float(case.size) * algorithm_input.cost_function.range_cost(0, case.size)
    greedy_solver_binary_segmentation: Solver = BinarySegmentation(algorithm_input=algorithm_input)
    while upper_penalization - lower_penalization > Constants.epsilon:
        penalization = (lower_penalization + upper_penalization) / 2.0
        algorithm_input.penalization = penalization
        solution_binary_segmentation: Solution = greedy_solver_binary_segmentation.solve()
        if len(solution_binary_segmentation.changepoints) > guessed_changepoints:
            lower_penalization = penalization
        else:
            upper_penalization = penalization
    return lower_penalization


def apply_elbow(changepoints_to_analyze: List[int], objective_values: List[float], threshold: float):
    guessed_changepoints = changepoints_to_analyze[0]
    # print([objective_values[k - 1] for k in changepoints_to_analyze])
    for k in range(1, len(changepoints_to_analyze) - 1):
        prev_i = changepoints_to_analyze[k - 1] - 1
        actual_i = changepoints_to_analyze[k] - 1
        next_i = changepoints_to_analyze[k + 1] - 1
        relatively_linear = float(objective_values[prev_i] - objective_values[actual_i]) < float(
            objective_values[actual_i] - objective_values[next_i]) * threshold
        decreasing_substantially = objective_values[prev_i] - objective_values[actual_i] > objective_values[changepoints_to_analyze[0]] * (threshold - 1.0)
        print(k, relatively_linear, decreasing_substantially)
        if decreasing_substantially:
            guessed_changepoints = changepoints_to_analyze[k - 1]
        if relatively_linear and not decreasing_substantially:
            break
    return guessed_changepoints


@dataclass
class PenalizationSelector:
    visualize: bool = False
    threshold: float = 1.01

    def select_penalization(self, case: Case) -> Tuple[float, int]:
        """
        Selects reasonable penalization values and maximum amount of
        changepoints for a given case.
        :param case: case to solve.
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

    def select_penalization(self, case: Case) -> Tuple[float, int]:
        """
        It runs dynamic programming suboptimal algorithm to find the solution for each amount of changepoint,
        and the uses binary search to find a suitable penalization for a binary segmentation algorithm to achieve
        such desired amount of changepoitns previously found with the elbow method.
        :param case: input case.

        :return: A tuple with a pair of suitable values for penalization and amount of changepoints.
        """

        objective_values, changepoints_values, solution_dynamic_programming, algorithm_input = obtain_solution_properties(case=case, account_penalization=True)
        amount_changepoints = len(objective_values)
        guessed_changepoints = apply_elbow(list(range(amount_changepoints)), objective_values, self.threshold)
        if self.visualize:
            visualize_elbow(case, solution_dynamic_programming, guessed_changepoints)
        return obtain_penalization_from_changepoints(case, algorithm_input, guessed_changepoints), guessed_changepoints


class SilhouettePenalizationSelector(PenalizationSelector):
    """
    It uses the solutions that run relatively quickly, even if they are suboptimal, in
    order to choose suitable values of total amount of changepoints and penalization that can
    be used as input to other algorithms.

    Once it has the solutions obtained for each amount of changepoints, it
    selects one that seems suitable in terms of the 'neighbouring silhouette method': https://en.wikipedia.org/wiki/Silhouette_(clustering)
    """
    aggregation_inside_range: Callable[[list[float]], float] = lambda self, values: sorted(values)[len(values) // 2]
    aggregation_signal: Callable[[list[float]], float] = lambda self, values: sorted(values)[len(values) // 2]

    def select_penalization(self, case: Case) -> Tuple[float, int]:
        objective_values, changepoints_values, solution_dynamic_programming, algorithm_input = obtain_solution_properties(case=case, account_penalization=False)
        amount_changepoints = len(objective_values)
        kernel = LaplaceKernel()
        aggregated_silhouette = []
        for k in range(1, amount_changepoints):
            silhouette = []
            from_indexes = [0] + list(reversed(changepoints_values[k]))
            to_indexes = list(reversed(changepoints_values[k])) + [case.size]
            for range_index in range(len(from_indexes)):
                from_index, to_index = from_indexes[range_index], to_indexes[range_index]
                for value in case.signal[from_index:to_index]:
                    inner_similarity = self.aggregation_inside_range(
                        [kernel.similarity(value, other_value) for other_value in case.signal[from_index:to_index]])
                    prev_similarity, next_similarity = 0.0, 0.0
                    if range_index > 0:
                        prev_from, prev_to = from_indexes[range_index - 1], to_indexes[range_index - 1]
                        prev_similarity = self.aggregation_inside_range(
                            [kernel.similarity(value, other_value) for other_value in case.signal[prev_from:prev_to]])
                    if range_index < len(from_indexes) - 1:
                        next_from, next_to = from_indexes[range_index + 1], to_indexes[range_index + 1]
                        next_similarity = self.aggregation_inside_range(
                            [kernel.similarity(value, other_value) for other_value in case.signal[next_from:next_to]])
                    neighbouring_similarity = max(prev_similarity, next_similarity)
                    silhouette.append((inner_similarity - neighbouring_similarity) / max(neighbouring_similarity, inner_similarity))
            aggregated_silhouette.append((self.aggregation_signal(silhouette), k))
        silhouette_candidates = [(value[0], objective_values[value[1]], value[1]) for value in aggregated_silhouette]
        max_silhouette = max([value[0] for value in silhouette_candidates])
        min_objective_value = min([value[1] for value in silhouette_candidates])
        # print(sorted(
        #     [((value[0] / max_silhouette) * (min_objective_value / value[1]) * math.exp(-value[2] / Constants.changepoints_bound), value[2]) for value in
        #      silhouette_candidates], reverse=True))
        guessed_changepoints = \
        max([((value[0] / max_silhouette) * (min_objective_value / value[1]) * math.exp(-value[2] / Constants.changepoints_bound), value[2]) for value in
             silhouette_candidates])[1]
        if self.visualize:
            visualize_silhouette(case, solution_dynamic_programming, [value[0] for value in aggregated_silhouette], guessed_changepoints)
        return obtain_penalization_from_changepoints(case, algorithm_input, guessed_changepoints), guessed_changepoints

    def with_aggregations(self, name_inside_range: str, name_signal: str) -> PenalizationSelector:
        aggregations = {'mean': lambda values: float(sum(values)) / float(len(values)), 'median': lambda values: sorted(values)[len(values) // 2],
                        'max': lambda values: float(max(values)), 'min': lambda values: float(min(values)),
                        'squared': lambda values: sum([float(x * x) for x in values]) / float(len(values)),
                        'p01': lambda values: sorted(values)[1 * len(values) // 100], 'p05': lambda values: sorted(values)[5 * len(values) // 100],
                        'p10': lambda values: sorted(values)[10 * len(values) // 100], 'p15': lambda values: sorted(values)[15 * len(values) // 100],
                        'p25': lambda values: sorted(values)[25 * len(values) // 100], 'p35': lambda values: sorted(values)[35 * len(values) // 100],
                        'p75': lambda values: sorted(values)[75 * len(values) // 100], 'p95': lambda values: sorted(values)[95 * len(values) // 100]}
        self.aggregation_inside_range = aggregations[name_inside_range]
        self.aggregation_signal = aggregations[name_signal]
        return self
