from cases.case import Case
from cost_functions.cost_function import KernelBasedCostFunction, GaussianCostFunction
from cost_functions.kernels import LaplaceKernel
from runner.run_utils import read_case, run_solution
from solution.algorithm_input import AlgorithmInput
from solution.binary_segmentation import BinarySegmentation
from solution.optimal_partition_penalization import OptimalPartition
from solution.solver import Solver


def main():
    """
    Runs a single case.
    :return: None.
    """
    case: Case = read_case('00_mean', 'random')
    cost_function: GaussianCostFunction = GaussianCostFunction()
    # cost_function: KernelBasedCostFunction = KernelBasedCostFunction()
    # cost_function.set_kernel(LaplaceKernel())
    # solver: Solver = BinarySegmentation(algorithm_input=AlgorithmInput(case=case, cost_function=cost_function))
    solver: Solver = OptimalPartition(algorithm_input=AlgorithmInput(case=case, cost_function=cost_function))
    run_solution([solver], [cost_function], case)


if __name__ == '__main__':
    main()
