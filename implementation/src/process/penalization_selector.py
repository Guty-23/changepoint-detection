import math
from datetime import datetime
from typing import Tuple

from cases.case import Case, ValueMetadata
from cost_functions.cost_function import KernelBasedCostFunction
from solution.algorithm_input import AlgorithmInput
from solution.solution import Solution
from solution.solver import Solver
from solution.suboptimal_partition_changepoints_in_state_divide_and_conquer_optimzation import DynamicProgrammingDivideAndConquer
from utils.constants import Constants
from visualization.visualization_script import visualize, visualize_elbow


def select_penalization(case: Case) -> Tuple[float, int]:
    """
    Selects reasonable penalization values and maximum amount of
    changepoints for a given case.
    :param case: case to solve
    :return: a pair with the penalization and max amount of changepoints.
    """

    changepoints_bound: int = min(500, math.floor(math.sqrt(case.size)))
    default_cost_function: KernelBasedCostFunction = KernelBasedCostFunction()
    algorithm_input = AlgorithmInput(case=case, cost_function=default_cost_function, max_amount_changepoints=changepoints_bound)
    algorithm_input.initialize()
    greedy_solver: Solver = DynamicProgrammingDivideAndConquer(algorithm_input=algorithm_input)
    solution: Solution = greedy_solver.solve()
    visualize_elbow(case, solution)
    return 0.0, 0


def aux_read_case(case_id: str, case_type: str = 'random') -> Case:
    case_path = Constants.real_path if case_type == 'real' else Constants.random_path + 'generated/'
    with open(case_path + case_id + '.in') as input_file:
        input_values = list(map(float, input_file.readline().split(',')))
        if case_type == 'real':
            input_metadata = list(map(
                lambda index_date_str_pair: ValueMetadata(index_date_str_pair[0],
                                                          datetime.strptime(index_date_str_pair[1].strip(), Constants.date_format)),
                enumerate(input_file.readline().split(','))))
        else:
            input_metadata = [ValueMetadata(index, Constants.no_date) for index in range(len(input_values))]
    return Case(name=case_id, size=len(input_values), signal=input_values, metadata=input_metadata, case_type=case_type)

if __name__ == '__main__':
    select_penalization(aux_read_case('00_mean', 'random'))

