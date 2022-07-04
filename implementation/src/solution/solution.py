from dataclasses import dataclass
from typing import List

from metrics.metrics import Metrics


@dataclass
class Solution:
    changepoints: List[int]
    metrics: Metrics
