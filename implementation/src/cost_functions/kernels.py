class Kernel:
    """
    Auxiliary kernel function used to measure similarity among two
    different values.
    """

    def simlarity(self, x: float, y: float) -> float:
        """
        Number between [0,1] that measures how similar
        two values are (close to 1 if they are similar,
        close to 0 if they are not).
        :param x: first value.
        :param y: second value.
        :return: real value that tries to capture how similar two values are.
        """
