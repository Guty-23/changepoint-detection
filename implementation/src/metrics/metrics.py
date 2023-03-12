from dataclasses import dataclass
from typing import List

from utils.constants import Constants


@dataclass
class Metrics:
    cost: float
    solver_used: str
    execution_time: float
    best_prefix: List[List[float]]
    correct_changepoints: int = Constants.no_data
    incorrect_changepoints: int = Constants.no_data
    not_found_changepoints: int = Constants.no_data
