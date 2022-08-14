from dataclasses import dataclass, field
from typing import List

from cost_functions.kernels import Kernel, LaplaceKernel
from utils.aux import accumulate, range_sum
from utils.constants import Constants

@dataclass
class CostFunction:
    """
    A function that associates a cost to a
    given range in the signal.
    """
    name: str = 'general_cost_function'

    def range_cost(self, start: int, end: int) -> float:
        """
        Cost associated to [start, end).
        :param start: begin of the range (inclusive).
        :param end: end of the range (exclusive).
        :return: real value with the associated cost to the range.
        """

    def precompute(self, signal: List[float]) -> None:
        """
        Performs any pre-computations needed to answer
        range cost queries efficiently.
        :param signal: problem input signal.
        :return: None.
        """


@dataclass
class GaussianCostFunction(CostFunction):
    """ Gaussian cost function. """
    name: str = 'gaussian'
    prefix_sum: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)
    prefix_sum_squares: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)

    def precompute(self, signal: List[float]) -> None:
        self.prefix_sum = accumulate(signal)
        self.prefix_sum_squares = accumulate(signal, lambda x: x ** 2)

    def range_cost(self, start: int, end: int) -> float:
        if start == end:
            return Constants.infinity
        inv_length = 1.0 / float(end - start)
        linear_sum_term = (inv_length ** 2) * (range_sum(self.prefix_sum, start, end) ** 2)
        square_sum_term = inv_length * range_sum(self.prefix_sum_squares, start, end)
        return square_sum_term - linear_sum_term


@dataclass
class ExponentialCostFunction(CostFunction):
    """ Exponential distribution cost function. """
    name: str = 'exponential'
    prefix_sum: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)

    def precompute(self, signal: List[float]) -> None:
        self.prefix_sum = accumulate(signal)

    def range_cost(self, start: int, end: int) -> float:
        range_value = max(range_sum(self.prefix_sum, start, end), Constants.epsilon)
        return float(end - start) / range_value


@dataclass
class KernelBasedCostFunction(CostFunction):
    """Kernel based cost function. """
    kernel: Kernel = LaplaceKernel()
    name: str = kernel.name
    prefix_sum_1d: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)
    prefix_sum_2d: List[List[float]] = field(default_factory=list, compare=False, hash=False, repr=False)

    def set_kernel(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.name = kernel.name

    def sum_submatrix(self, start: int, end: int):
        return self.prefix_sum_2d[end][end] - \
               self.prefix_sum_2d[start][end] - \
               self.prefix_sum_2d[end][start] + \
               self.prefix_sum_2d[start][start]

    def precompute(self, signal: List[float]) -> None:
        self.prefix_sum_1d = accumulate([self.kernel.similarity(x, x) for x in signal])
        self.prefix_sum_2d = [[0.0 for _ in range(len(signal) + 1)]] + \
                             [[0.0] + [self.kernel.similarity(x, y) for y in signal] for x in signal]

        for i in range(len(signal)):
            for j in range(len(signal)):
                self.prefix_sum_2d[i + 1][j + 1] += self.prefix_sum_2d[i][j + 1] + \
                                                    self.prefix_sum_2d[i + 1][j] - \
                                                    self.prefix_sum_2d[i][j]

    def range_cost(self, start: int, end: int) -> float:
        return Constants.infinity if start == end else \
            (self.prefix_sum_1d[end] - self.prefix_sum_1d[start]) - (1.0 / float(end - start) * self.sum_submatrix(start, end))
