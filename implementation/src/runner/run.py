from cases.case import Case
from cost_functions.cost_function import KernelBasedCostFunction, GaussianCostFunction
from cost_functions.kernels import LaplaceKernel
from runner.run_utils import read_case, run_solution, read_output
from solution.algorithm_input import AlgorithmInput
from solution.binary_segmentation import BinarySegmentation
from solution.optimal_partition_changepoints_in_state import DynamicProgrammingChangepointsInState
from solution.optimal_partition_changepoints_in_state_pruned import DynamicProgrammingChangepointsInStatePruned
from solution.optimal_partition_penalization import DynamicProgrammingPenalization
from solution.optimal_partition_penalization_pruned import DynamicProgrammingPenalizationPruned
from solution.solution import Solution
from solution.solver import Solver
from solution.suboptimal_partition_changepoints_in_state_divide_and_conquer_optimzation import DynamicProgrammingDivideAndConquer
from visualization.visualization_script import visualize


def run_case(visualize_case: bool = False) -> None:
    """
    Runs a single case.
    :return: None.
    """
    case: Case = read_case('00_mean', 'random')

    # cost_function: GaussianCostFunction = GaussianCostFunction()
    cost_function: KernelBasedCostFunction = KernelBasedCostFunction()

    algorithm_input = AlgorithmInput(case=case, cost_function=cost_function).initialize()

    # solver: Solver = BinarySegmentation(algorithm_input=algorithm_input)
    # solver: Solver = DynamicProgrammingPenalization(algorithm_input=algorithm_input)
    # solver: Solver = DynamicProgrammingPenalizationPruned(algorithm_input=algorithm_input)
    # solver: Solver = DynamicProgrammingChangepointsInState(algorithm_input=algorithm_input)
    # solver: Solver = DynamicProgrammingChangepointsInStatePruned(algorithm_input=algorithm_input)
    solver: Solver = DynamicProgrammingDivideAndConquer(algorithm_input=algorithm_input)

    run_solution([solver], [cost_function], case)
    solution: Solution = read_output(case.name, case.case_type, solver.name)
    if visualize_case:
        visualize(case, solution)


if __name__ == '__main__':
    run_case(visualize_case=True)
