import math
from dataclasses import dataclass

from utils.constants import Constants


@dataclass
class Kernel:
    """
    Auxiliary kernel function used to measure similarity among two
    different values.
    """
    bandwith: float = Constants.kernel_bandwidth
    name: str = 'general_kernel'

    def similarity(self, x: float, y: float) -> float:
        """
        Number between [0,1] that measures how similar
        two values are (close to 1 if they are similar,
        close to 0 if they are not).
        :param x: first value.
        :param y: second value.
        :return: real value that tries to capture how similar two values are.
        """


class GaussianKernel(Kernel):
    """ Gaussian Kernel. """
    name: str = 'gaussian_kernel'

    def similarity(self, x: float, y: float) -> float:
        return math.exp(-(abs(x - y) ** 2) / (2 * (self.bandwith ** 2)))


class LaplaceKernel(Kernel):
    """ Laplace Kernel. """
    name: str = 'laplace_kernel'

    def similarity(self, x: float, y: float) -> float:
        return math.exp(-abs(x - y) / self.bandwith)
