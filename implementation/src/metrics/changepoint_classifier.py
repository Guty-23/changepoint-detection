from typing import List, Set, Tuple

from utils.constants import Constants


def real_changepoints(real_changepoints: List[int], found_changepoints: List[int]) -> Tuple[Set[int], Set[int]]:
    """
    It calculates the correctly found and not found changepoints.
    :param real_changepoints:
    :param found_changepoints:
    :return: A tuple with two sets. The first one are the real changepoints not found, the second one are the changepoints correctly found.
    """
    real_correctly_found = set()
    found_correctly = set()
    for changepoint in found_changepoints:
        for real_changepoint in real_changepoints:
            if real_changepoint not in real_correctly_found and abs(changepoint - real_changepoint) <= Constants.window_threshold:
                real_correctly_found.add(real_changepoint)
                found_correctly.add(changepoint)
                break
    return set(real_changepoints).difference(real_correctly_found), found_correctly
