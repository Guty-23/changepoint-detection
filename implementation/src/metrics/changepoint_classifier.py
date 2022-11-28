from typing import List, Set

THRESHOLD = 10


def real_changepoints(real_changepoints: List[int], found_changepoints: List[int]) -> Set[int]:
    real_correctly_found = set()
    found_correctly = set()
    for changepoint in found_changepoints:
        for real_changepoint in real_changepoints:
            if real_changepoint not in real_correctly_found and abs(changepoint - real_changepoint) <= THRESHOLD:
                real_correctly_found.add(real_changepoint)
                found_correctly.add(changepoint)
                break
    return real_correctly_found, found_correctly
