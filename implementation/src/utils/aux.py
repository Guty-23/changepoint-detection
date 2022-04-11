from typing import List, Callable

import numpy as np


def accumulate(signal: List[float], mapping: Callable = lambda x: x) -> List[float]:
    return [float(x) for x in np.insert(np.cumsum(map(mapping, signal)), 0, 0)]
