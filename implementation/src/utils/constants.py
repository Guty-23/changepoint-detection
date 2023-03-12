import datetime
import os
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Constants:
    """
    This represents an input where
    existent changepoints are to be located
    """
    seed: int = 1
    cases_per_type: int = 10
    mean_limit: int = 15
    batch_size: int = 1000
    std_limit: int = 8
    min_days: int = 2
    minutes_in_a_day: int = 24 * 60
    kernel_bandwidth: float = 1e3
    window: int = 30
    epsilon: float = 1e-6
    infinity: float = 1e12
    changepoints_bound: int = 250
    window_threshold: int = 10
    project_root_path: str = os.path.dirname(os.path.abspath(__file__)) + '/../../'
    random_path: str = project_root_path + 'resources/cases/random/'
    real_path: str = project_root_path + 'resources/cases/real/'
    output_path: str = project_root_path + 'output/cases/'
    date_format: str = '%Y-%m-%d %H:%M'
    no_date: datetime.datetime = datetime.datetime(year=1970, month=1, day=1)
    no_data: datetime.datetime = -1
    metrics_columns: str = (
    'name', 'size', 'cost_function', 'solver', 'changepoints', 'cost', 'execution_time', 'right_changepoints', 'wrong_changepoints', 'not_found_changepoints')
