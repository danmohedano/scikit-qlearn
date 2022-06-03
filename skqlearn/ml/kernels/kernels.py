import numpy as np


class SqueezingKernel:
    """SqueezingKernel :cite:`hilbert2019`"""
    def __init__(self, c: float):
        """

        Args:
            c (float): Strenght of the squeezing.
        """
        self.c = c

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculation of the Gram matrix between the two sets of vectors.

        Args:
            x (numpy.ndarray of shape (n_samples_1, n_features)): First input.
            y (numpy.ndarray of shape (n_samples_2, n_features)): Second input.

        Returns:
            numpy.ndarray of shape (n_samples_1, n_samples_2):
                Resulting Gram matrix.

        Raises:
            ValueError: if dimensions mismatch.
        """
        if x.shape[1] != y.shape[1]:
            raise ValueError('Invalid input dimensions. Expected vectors with '
                             'same amount of features, '
                             f'got {x.shape} and {y.shape} instead.')

        gram = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                inner = 1
                for k in range(x.shape[1]):
                    inner *= self._inner(x[i, k], y[j, k])

                gram[i, j] = abs(inner) ** 2

        return gram

    def _inner(self, a: float, b: float) -> complex:
        """Calculation of the inner product operation.

        Args:
            a (float): First input.
            b (float): Second input.

        Returns:
            complex:
                Result.
        """
        num = (1 / np.cosh(self.c) ** 2)
        denom = 1 - ((np.e ** (1j * (b - a))) * np.tanh(self.c) ** 2)
        return np.sqrt(num / denom)