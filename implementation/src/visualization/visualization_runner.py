from cases.case import Case
from runner.run_utils import read_case, read_output
from solution.solution import Solution
from visualization.visualization_script import visualize


def main():
    case_name = '00_mean'
    case_type = 'random'
    solver_used = 'optimal_partition_penalization'
    case: Case = read_case(case_name, case_type)
    solution: Solution = read_output(case_name, case_type, solver_used)
    visualize(case, solution)


if __name__ == '__main__':
    main()
