import os
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Constants:
    """
    This represents an input where
    existent changepoints are to be located
    """
    seed = 23
    cases_per_type = 10
    mean_limit = 15
    batch_size = 1000
    std_limit = 8
    min_days = 2
    minutes_in_a_day = 24 * 60
    project_root_path = os.path.dirname(os.path.abspath(__file__)) + '/../../'
    random_path = project_root_path + 'resources/cases/random/'
    real_path = project_root_path + 'resources/cases/real/'
