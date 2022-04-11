import os
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Constants:
    """
    This represents an input where
    existent changepoints are to be located
    """
    seed: int = 23
    cases_per_type: int = 10
    mean_limit: int = 15
    batch_size: int = 1000
    std_limit: int = 8
    min_days: int = 2
    minutes_in_a_day: int = 24 * 60
    kernel_bandwidth: float = 1e-3
    project_root_path: str = os.path.dirname(os.path.abspath(__file__)) + '/../../'
    random_path: str = project_root_path + 'resources/cases/random/'
    real_path: str = project_root_path + 'resources/cases/real/'
