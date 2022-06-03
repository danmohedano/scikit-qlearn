import numpy as np


class SqueezingKernel:
    r"""Kernel inspired by squeezed vacuum states :cite:`hilbert2019`

    It is based on the definition of a *squeezed vacuum state* of the
    electromagnetic field as:

    .. math::
       \left|z\right> = \frac{1}{\sqrt{\cosh(r)}}
       \sum_{n=0}^{\infty}\frac{\sqrt{(2n)!}}{2^n n!}
       (-e^{i\varphi}\tanh(r))\left|2n\right>

    The notation :math:`\left|z\right> = \left|(r, \varphi)\right>` is also
    used, where :math:`z=re^{i\varphi}`.

    :math:`x\rightarrow\left|\phi(x)\right>=\left|(c,x)\right>` can be
    interpreted as a feature map for :math:`x\in\mathbb{R}`. The hyperparameter
    :math:`c` determines the strength of the squeezing. This can be extended
    to vectors of the form
    :math:`\boldsymbol{x}=(x_1, ..., x_N)^T\in\mathbb{R}^N`,
    defining the feature map as:

    .. math::
       \phi: \boldsymbol{x} \rightarrow \left|(c,\boldsymbol{x})\right>=
       \left|(c,x_1)\right> \bigotimes ... \bigotimes\left|(c,x_N)\right>

    It therefore defines the following kernel:

    .. math::
       k(\boldsymbol{x}, \boldsymbol{x}')=\prod_{i=1}^{N} \left<(c,x_i)|
       (c, x_i')\right>=\prod_{i=1}^N\sqrt{\frac{\text{sech }c\text{ sech }c}
       {1-e^{i(x_i'-x_i)}\tanh c \tanh c}}

    This can be evaluated classically by taking the absolute square of the
    inner product.

    """
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