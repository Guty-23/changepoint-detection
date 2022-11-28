import metrics.changepoint_classifier
from cases.case import Case
from cost_functions.cost_function import KernelBasedCostFunction, GaussianCostFunction, ExponentialCostFunction, CostFunction
from cost_functions.kernels import LaplaceKernel
from process.penalization_selector import PenalizationSelector, SilhouettePenalizationSelector
from runner.run_utils import read_case, run_solution, read_output
from solution.algorithm_input import AlgorithmInput
from solution.binary_segmentation import BinarySegmentation
from solution.optimal_partition_changepoints_in_state import DynamicProgrammingChangepointsInState
from solution.optimal_partition_changepoints_in_state_pruned import DynamicProgrammingChangepointsInStatePruned
from solution.optimal_partition_penalization import DynamicProgrammingPenalization
from solution.optimal_partition_penalization_pruned import DynamicProgrammingPenalizationPruned
from solution.solution import Solution
from solution.suboptimal_partition_changepoints_in_state_divide_and_conquer_optimzation import DynamicProgrammingDivideAndConquer
from utils.constants import Constants
from visualization.visualization_script import visualize_solution


def run_case(visualize_case: bool = False) -> None:
    """
    Runs a single case.
    :return: None.
    """
    case: Case = read_case('01_mean', 'random')
    # cost_function: CostFunction = GaussianCostFunction()
    cost_function: CostFunction = KernelBasedCostFunction()
    # cost_function: CostFunction = ExponentialCostFunction()

    algorithm_input = AlgorithmInput(case=case, cost_function=cost_function).initialize()

    solver_list = [BinarySegmentation(algorithm_input=algorithm_input),
                   DynamicProgrammingPenalization(algorithm_input=algorithm_input),
                   DynamicProgrammingPenalizationPruned(algorithm_input=algorithm_input),
                   DynamicProgrammingChangepointsInState(algorithm_input=algorithm_input),
                   DynamicProgrammingChangepointsInStatePruned(algorithm_input=algorithm_input),
                   DynamicProgrammingDivideAndConquer(algorithm_input=algorithm_input)]

    penalization_selector: PenalizationSelector = SilhouettePenalizationSelector(visualize=visualize_case).with_aggregations('min', 'median')

    run_solution(solver_list, [cost_function], case, penalization_selector)
    for solver in solver_list:
        solution: Solution = read_output(case.name, case.case_type, solver.name)
        with open(Constants.random_path + 'solutions/' + case.name + '.out', 'r') as real_changepoitns_file:
            real_changepoints = list(map(int, real_changepoitns_file.readline().replace('\n', '').split(',')))
            real_correct_changepoints, found_correct_changepoints = metrics.changepoint_classifier.real_changepoints(real_changepoints, solution.changepoints)
            print(solver.name.ljust(50), len(solution.changepoints), solution.metrics.cost)
            if visualize_case:
                visualize_solution(case, solution, found_correct_changepoints)


if __name__ == '__main__':
    run_case(visualize_case=True)
