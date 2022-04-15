from typing import List, Callable

import numpy as np


def accumulate(signal: List[float], mapping: Callable = lambda x: x) -> List[float]:
    """
    It accumulates the signal after applying the mapping element-wise.
    :param signal: input signal.
    :param mapping: function to be applied to each signal value.
    :return: accumulated signal.
    """
    return [float(x) for x in np.insert(np.cumsum(map(mapping, signal)), 0, 0)]


def range_sum(accumulated_signal: List[float], start: int, end: int) -> float:
    """
    It returns the sum in the range by calculating the difference in the accumulated array.
    :param accumulated_signal: input signal that has already been accumulated.
    :param start: beginning of the range (inclusive).
    :param end: final index of the range (exclusive).
    :return: value of the sum in the given range.
    """
    return accumulated_signal[end] - accumulated_signal[start]
