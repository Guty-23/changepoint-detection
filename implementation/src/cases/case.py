from dataclasses import dataclass, field
from typing import List

import numpy as np

from utils.constants import Constants


@dataclass(frozen=True, order=True)
class Case:
    """
    This represents an input where existent changepoints
    are not known and are yet to be estimated
    """
    size: int = 0
    name: str = ''
    signal: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)


@dataclass(order=True, frozen=True)
class CaseParameters:
    """
    This represents all the information needed to create
    a specific case.
    """
    # Length of the signal to be generated.
    size: int = 0
    # Amount of changepoints to be generated.
    changepoints: int = 0
    # Fixed mean for the Gaussian samples.
    mu: int = 0
    # Fixed standard deviation for the Gaussian samples.
    sigma: int = 1
    # Lower bound to be used in the uniform sample of a new mean for the Gaussian.
    mu_low: int = 0
    # Upper bound to be used in the uniform sample of a new mean for the Gaussian.
    mu_high: int = 0
    # Lower bound to be used in the uniform sample of a new standard deviation for the Gaussian.
    sigma_low: int = 0
    # Upper bound to be used in the uniform sample of a new standard deviation for the Gaussian.
    sigma_high: int = 0
    # Lower bound to be used in the uniform sample of a new mean for the Exponential distribution.
    lambda_low: int = 1
    # Upper bound to be used in the uniform sample of a new mean for the Exponential distribution.
    lambda_high: int = 1

    # Fixed random number generator.
    rng: np.random.RandomState = np.random.RandomState(Constants.seed)
