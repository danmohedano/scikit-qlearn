from .base_encoding import Encoding
import numpy as np


class AngleEncoding(Encoding):
    r"""Angle encoding method. :cite:`LaRose_2020`

    In angle encoding, each component of the input vector
    :math:`\boldsymbol{x} \in \mathbb{R}^N` is mapped to a qubit, defining the
    encoding feature map as:

    .. math::
       \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
       \bigotimes_{i=1}^{N}\cos{x_i}\left|0\right>+\sin{x_i}\left|1\right>

    Because of the encoding feature map, the resulting quantum state is
    correctly normalized and therefore valid, as :math:`\cos{x}^2+\sin{x}^2=1`.

    The kernel defined by the inner product is a cosine kernel:

    .. math::
       k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
       \psi_{\boldsymbol{x'}}\right> = \prod_{i=1}^{N}\sin{x_i}\sin{x'_i} +
       \cos{x_i}\cos{x'_i}=\prod_{i=1}^{N}\cos{(x_i-x'_i)}
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of angle encoding to the input.

        In angle encoding, each feature of the input vector is encoded into a
        qubit.

        Args:
            x (numpy.ndarray of shape (n_features,)): Input vector.

        Returns:
            numpy.ndarray:
                Quantum state described with an amplitude vector.

        Raises:
            ValueError: When an invalid input is provided.

        Examples:
            >>> a = np.array([0.0])
            >>> AngleEncoding().encoding(a)
            array([1., 0.])

            >>> a = np.array([np.pi / 2, 0])
            >>> AngleEncoding().encoding(a)
            array([6.123234e-17, 0.000000e+00, 1.000000e+00, 0.000000e+00])
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(f'Invalid input type provided. Expected '
                             f'np.ndarray, got {type(x)} instead.')
        elif len(x.shape) == 1:
            return self._encoding_single(x)
        elif len(x.shape) == 2:
            return self._encoding_dataset(x)
        else:
            raise ValueError(f'Invalid input shape provided. Expected 1D or 2D'
                             f', got {x.shape} instead.')

    def _encoding_single(self, x: np.ndarray) -> np.ndarray:
        """Application of angle encoding to a single sample.

        Args:
            x (numpy.ndarray of shape(n_features,)): Input sample.

        Returns:
            numpy.ndarray:
                Quantum state described with an amplitude vector.
        """
        # Encoding of each feature into a qubit and use of Kronecker product to
        # build the amplitude vector.
        state = 1
        for i in range(x.shape[0]):
            qubit = np.array([np.cos(x[i]), np.sin(x[i])])
            state = np.kron(state, qubit)

        return state

    def _encoding_dataset(self, x: np.ndarray) -> np.ndarray:
        """Application of angle encoding to a dataset by concatenation.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Input dataset.

        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector
                constructed by concatenating the quantum states for each
                sample.
        """
        # Calculate the size of a single state
        vector_size = 2 ** x.shape[1]

        states = np.zeros(vector_size * x.shape[0])
        for i in range(x.shape[0]):
            states[i * vector_size: (i + 1) * vector_size] = \
                self._encoding_single(x[i, :])

        # Normalization of the concatenated vectors
        states /= np.sqrt(x.shape[0])

        return states
