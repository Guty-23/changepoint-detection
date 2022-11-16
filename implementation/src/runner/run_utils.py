import os
from datetime import datetime
from typing import List, TextIO

import pandas

from cases.case import Case, ValueMetadata
from cost_functions.cost_function import CostFunction
from metrics.metrics import Metrics
from process.penalization_selector import ElbowPenalizationSelector, SilhouettePenalizationSelector, PenalizationSelector
from solution.algorithm_input import AlgorithmInput
from solution.solution import Solution
from solution.solver import Solver
from utils.constants import Constants


def write_metrics(algorithm_input: AlgorithmInput, solver: Solver, metrics_file: TextIO, amount_changepoints: int, solution_metrics: Metrics):
    metrics_file.write(','.join(map(str, [
        algorithm_input.case.name,
        algorithm_input.case.size,
        algorithm_input.cost_function.name,
        solver.name,
        amount_changepoints,
        solution_metrics.cost])) + '\n')


def read_case(case_id: str, case_type: str = 'random') -> Case:
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


def read_output(case_id: str, case_type: str = 'random', solver_used='binary_segmentation') -> Solution:
    solution_file_path = '/'.join([Constants.output_path, case_type, case_id + '_' + solver_used + '.out'])
    metrics_file_path = '/'.join([Constants.output_path, case_type, case_id + '.metrics'])
    with open(solution_file_path, 'r') as output_file:
        changepoints = list(map(int, output_file.readline().split(',')))
    metrics_df = pandas.read_csv(metrics_file_path)
    cost = float(metrics_df[metrics_df['solver'] == solver_used]['cost'].iloc[0])
    return Solution(changepoints, Metrics(cost, solver_used, []))


def run_solution(solvers: List[Solver], cost_functions: List[CostFunction], case: Case, penalization_selector: PenalizationSelector) -> None:
    penalization, max_amount_changepoints = penalization_selector.select_penalization(case)
    for cost_function in cost_functions:
        algorithm_input = AlgorithmInput(
            case=case,
            cost_function=cost_function,
            penalization=penalization,
            max_amount_changepoints=max_amount_changepoints)
        path = Constants.output_path + algorithm_input.case.case_type + '/'
        os.makedirs(path, exist_ok=True)
        with open(path + algorithm_input.case.name + '.metrics', 'w') as metrics_file:
            metrics_file.write(','.join(list(Constants.metrics_columns)) + '\n')
            for solver in solvers:
                solver.set_input(algorithm_input)
                solution = solver.solve()
                write_metrics(algorithm_input, solver, metrics_file, len(solution.changepoints), solution.metrics)
                with open(path + algorithm_input.case.name + '_' + solver.name + '.out', 'w') as output_file:
                    output_file.write(','.join(list(map(str, sorted(solution.changepoints)))) + '\n')
