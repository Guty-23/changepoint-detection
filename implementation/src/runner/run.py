from cases.case import Case
from cost_functions.cost_function import KernelBasedCostFunction, GaussianCostFunction
from cost_functions.kernels import LaplaceKernel
from runner.run_utils import read_case, run_solution, read_output
from solution.algorithm_input import AlgorithmInput
from solution.binary_segmentation import BinarySegmentation
from solution.optimal_partition_changepoints_in_state import OptimalPartitionChangepointsInState
from solution.optimal_partition_changepoints_in_state_pruned import OptimalPartitionChangepointsInStatePruned
from solution.optimal_partition_penalization import OptimalPartition
from solution.optimal_partition_penalization_pruned import OptimalPartitionPruned
from solution.solution import Solution
from solution.solver import Solver
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
    # solver: Solver = OptimalPartition(algorithm_input=algorithm_input)
    # solver: Solver = OptimalPartitionPruned(algorithm_input=algorithm_input)
    # solver: Solver = OptimalPartitionChangepointsInState(algorithm_input=algorithm_input)
    solver: Solver = OptimalPartitionChangepointsInStatePruned(algorithm_input=algorithm_input)

    run_solution([solver], [cost_function], case)
    solution: Solution = read_output(case.name, case.case_type, solver.name)
    if visualize_case:
        visualize(case, solution)


if __name__ == '__main__':
    run_case(visualize_case=True)
