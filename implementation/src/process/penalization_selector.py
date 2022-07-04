from typing import Tuple

from cases.case import Case


def select_penalization(case: Case) -> Tuple[float, int]:
    """
    Selects reasonable penalization values and maximum amount of
    changepoints for a given case.
    :param case:
    :return:
    """
    if case.case_type == 'real':
        return 7.5, 50
    else:
        return 0.1, 50
