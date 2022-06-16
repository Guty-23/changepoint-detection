from cases.case import Case
from cost_functions.cost_function import CostFunction, KernelBasedCostFunction
from runner.run_utils import read_case, run_solution
from solution.algorithm_input import AlgorithmInput
from solution.binary_segmentation import BinarySegmentation
from solution.solver import Solver


def main():
    case: Case = read_case('00_real', 'real')
    cost_function: CostFunction = KernelBasedCostFunction()
    solver: Solver = BinarySegmentation(algorithm_input=AlgorithmInput(case=case, cost_function=cost_function))
    run_solution([solver], [cost_function], case)


if __name__ == '__main__':
    main()
