from typing import Tuple

from cases.case import Case


def select_penalization(case: Case) -> Tuple[float, int]:
    return 10.0, 50
