from .base_encoding import Encoding
from typing import Union
import numpy as np


class BasisEncoding(Encoding):
    r"""Basis encoding method. :cite:`schuld2018supervised`

    In basis encoding, each classical bit is mapped into a single qubit,
    defining the encoding feature map as:

    .. math::
       \phi:i\rightarrow \left|i\right>

    Therefore, the kernel defined by the inner product is:

    .. math::
       k(i, j) = \left<\phi(i)|\phi(j)\right> = \left<i|j\right> = 
       \delta_{ij}

    With :math:`\delta` being the Kronecker delta.

    The encoding also permits encoding an entire dataset of binary strings
    :math:`\mathcal{D}=\{\boldsymbol{x}^1,...,\boldsymbol{x}^m\}` together as:

    .. math::
       \left|\mathcal{D}\right> = \frac{1}{\sqrt{M}}\sum_{m=1}^{M}
       \left|\boldsymbol{x}^m\right>
    """

    def encoding(self, x: Union[int, np.ndarray]) -> np.ndarray:
        """Application of basis encoding to the input.

        In basis encoding, each classical bit is mapped into a qubit.

        Args:
            x (int or numpy.ndarray): Either a single sample (int) or a dataset
                of shape (n_samples,).

        Returns:
            numpy.ndarray:
                Amplitude vector describing the input encoded into a
                quantum state. If a dataset is provided, the quantum state will
                be a superposition of the encodings of every sample in the
                dataset.

        Raises:
            ValueError: When an invalid input type is provided.

        Examples:
            >>> a = 1
            >>> BasisEncoding().encoding(a)
            array([0., 1.])

            >>> a = np.array([0, 1])
            >>> BasisEncoding().encoding(a)
            array([0.70710678, 0.70710678])
        """
        if isinstance(x, int) and x >= 0:
            return self._encoding_single(x)
        elif isinstance(x, np.ndarray) and len(x.shape) == 1:
            return self._encoding_dataset(x)
        else:
            raise ValueError('Invalid input type. Expected positive integer or'
                             f' np.ndarray, got {type(x)} instead.')

    def _encoding_single(self, x: int, size: int = -1) -> np.ndarray:
        """Application of basis encoding to a single input.

        Args:
            x (int): Input value.
            size (int): Size (in n_amplitudes) of the desired state. Must be a
                power of 2. If none is provided, then the size will be
                determined by the next larger power of 2.

        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector.
        """
        if size == -1:
            # Calculate the closest larger power of 2 (equivalent 2 calculating
            # the amount of qubits necessary and the size of the amplitude
            # vector needed to represent that state)
            if x < 2:
                size = 2
            else:
                size = int(2 ** np.ceil(np.log2(x + 1)))

        state = np.zeros(size)
        state[x] = 1.0

        return state

    def _encoding_dataset(self, x: np.ndarray) -> np.ndarray:
        """Application of basis encoding to a dataset.

        Args:
            x (numpy.ndarray of shape (n_samples,)): Input dataset.

        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector
                representing a superposition of all states in the dataset.

        Raises:
            ValueError: When an invalid input is provided.
        """
        if np.amin(x) < 0:
            raise ValueError('Invalid input provided.')

        max_data = np.amax(x)
        size = max(int(2 ** np.ceil(np.log2(max_data + 1))), 2)
        state = np.zeros(size)

        for i in range(x.shape[0]):
            # Addition of every state to create superposition
            state += self._encoding_single(x[i], size)

        # Normalization of the state
        state /= np.sqrt(x.shape[0])

        return state
