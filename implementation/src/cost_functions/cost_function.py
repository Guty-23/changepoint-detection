from dataclasses import dataclass, field
from typing import List

from utils.aux import accumulate


class CostFunction:
    """
    A function that associates a cost to a
    given range in the signal.
    """

    def range_cost(self, i: int, j: int) -> float:
        """
        Cost associated to [i,j).
        :param i: begin of the range (inclusive).
        :param j: end of the range (exclusive).
        :return: real value with the associated cost to the range.
        """

    def precompute(self, signal: List[int]) -> None:
        """
        Performs any precomuptations needed to answer
        range cost queries efficiently.
        :param signal: problem input signal.
        :return: None.
        """


@dataclass(order=True)
class GaussianCostFunction(CostFunction):
    """ Gaussian cost function. """
    prefix_sum: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)
    prefix_sum_squares: List[float] = field(default_factory=list, compare=False, hash=False, repr=False)

    def precompute(self, signal: List[int]) -> None:
        self.prefix_sum = accumulate(signal)
        self.prefix_sum_squares = accumulate(signal, lambda x: x ** 2)

    def range_cost(self, i: int, j: int) -> float:
        inv_length = 1.0 / float(j-i)
        linear_term = inv_length * (self.prefix_sum[j] - self.prefix_sum[i])
        square_term = (inv_length ** 2) * (self.prefix_sum_squares[j] - self.prefix_sum_squares[i])
        return linear_term + square_term
