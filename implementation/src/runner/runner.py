import os
from datetime import datetime
from typing import List

from cases.case import Case, ValueMetadata
from cost_functions.cost_function import CostFunction
from process.penalization_selector import select_penalization
from solution.algorithm_input import AlgorithmInput
from solution.solver import Solver
from utils.constants import Constants


class Runner:
    def read_case(self, case_id: str, case_type: str = 'random') -> Case:
        case_path = Constants.real_path if case_type == 'real' else Constants.random_path
        with open(case_path + case_id + '.in') as input_file:
            input_values = list(map(float, input_file.readline().split(',')))
            if case_type == 'real':
                input_metadata = list(map(lambda date_str: ValueMetadata(datetime.strptime(date_str, Constants.date_format)),
                                          input_file.readline().split(',')))
        return Case(name=case_id, size=len(input_values), signal=input_values, metadata=input_metadata, case_type=case_type)

    def run_solution(self, solvers: List[Solver], cost_functions: List[CostFunction], case: Case):
        penalization, max_amount_changepoints = select_penalization(case)
        for cost_function in cost_functions:
            algorithm_input = AlgorithmInput(
                case=case,
                cost_function=cost_function,
                penalization=penalization,
                max_amount_changepoints=max_amount_changepoints)
            for solver in solvers:
                solver.set_input(algorithm_input)
                changepoints, total_cost = solver.solve()
                path = Constants.output_path + algorithm_input.case.case_type
                os.makedirs(path, exist_ok=True)
                with open(path + '/' + algorithm_input.case.name + '.out') as output_file:
                    output_file.write(','.join(list(map(str, changepoints))) + '\n')
                columns = ['name', 'size', 'cost_function', 'solver']
                with open(path + '/' + algorithm_input.case.name + '.metrics') as metrics_file:
                    metrics_file.write()
