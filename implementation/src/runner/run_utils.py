import os
from datetime import datetime
from typing import List, TextIO

from cases.case import Case, ValueMetadata
from cost_functions.cost_function import CostFunction
from process.penalization_selector import select_penalization
from solution.algorithm_input import AlgorithmInput
from solution.solver import Solver
from utils.constants import Constants


def write_metrics(algorithm_input: AlgorithmInput, solver: Solver, metrics_file: TextIO, amount_changepoints: int, total_cost: float):
    metrics_file.write(','.join(map(str, [
        algorithm_input.case.name,
        algorithm_input.case.size,
        algorithm_input.cost_function.name,
        solver.name,
        amount_changepoints,
        total_cost])))


def read_case(case_id: str, case_type: str = 'random') -> Case:
    case_path = Constants.real_path if case_type == 'real' else Constants.random_path
    with open(case_path + case_id + '.in') as input_file:
        input_values = list(map(float, input_file.readline().split(',')))
        if case_type == 'real':
            input_metadata = list(map(lambda date_str: ValueMetadata(datetime.strptime(date_str.strip(), Constants.date_format)),
                                      input_file.readline().split(',')))
    return Case(name=case_id, size=len(input_values), signal=input_values, metadata=input_metadata, case_type=case_type)


def run_solution(solvers: List[Solver], cost_functions: List[CostFunction], case: Case):
    penalization, max_amount_changepoints = select_penalization(case)
    for cost_function in cost_functions:
        algorithm_input = AlgorithmInput(
            case=case,
            cost_function=cost_function,
            penalization=penalization,
            max_amount_changepoints=max_amount_changepoints)
        path = Constants.output_path + algorithm_input.case.case_type + '/' + algorithm_input.case.name
        os.makedirs(path, exist_ok=True)
        with open(path + '.metrics', 'w') as metrics_file:
            metrics_file.write(','.join(list(Constants.metrics_columns)) + '\n')
            for solver in solvers:
                solver.set_input(algorithm_input)
                changepoints, total_cost = solver.solve()
                write_metrics(algorithm_input, solver, metrics_file, len(changepoints), total_cost)
                with open(path + '.out', 'w') as output_file:
                    output_file.write(','.join(list(map(str, changepoints))) + '\n')
