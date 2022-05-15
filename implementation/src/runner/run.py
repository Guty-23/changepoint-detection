from datetime import datetime

from cases.case import Case, ValueMetadata
from utils.constants import Constants


class Runner:
    def read_case(self, case_id: str, case_type: str = 'random') -> Case:
        case_path = Constants.real_path if case_type == 'real' else Constants.random_path
        with open(case_path + case_id + '.in') as input_file:
            input_values = list(map(float, input_file.readline().split(',')))
            if case_type == 'real':
                input_metadata = list(map(lambda date_str: ValueMetadata(datetime.strptime(date_str, Constants.date_format)),
                                          input_file.readline().split(',')))
        return Case(name=case_id, size=len(input_values), signal=input_values, metadata=input_metadata)


