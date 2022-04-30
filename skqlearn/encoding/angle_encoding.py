from .encoding import Encoding
import numpy as np


class AngleEncoding(Encoding):
    r"""Angle encoding method. :cite:`LaRose_2020`

    In angle encoding, each component/feature of the input vector
    :math:`\boldsymbol{x} \in \mathbb{R}^N` is mapped to a qubit, defining the
    encoding feature map as:

    .. math::
       \phi:\boldsymbol{x}\rightarrow\ket{\psi_\boldsymbol{x}}=
       \bigotimes_{i=1}^{N}\cos{x_i}\ket{0}+\sin{x_i}\ket{1}

    Because of the encoding feature map, the resulting quantum state is
    correctly normalized and therefore valid, as :math:`\cos{x}^2+\sin{x}^2=1`.

    The kernel defined by the inner product is a cosine kernel:

    .. math::
       k(\boldsymbol{x}, \boldsymbol{x'}) = \braket{\psi_{\boldsymbol{x}}|
       \psi_{\boldsymbol{x'}}} = \prod_{i=1}^{N}\sin{x_i}\sin{x'_i} + \cos{x_i}
       \cos{x'_i}=\prod_{i=1}^{N}\cos{(x_i-x'_i)}
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of angle encoding to the input.

        In angle encoding, each feature of the input vector is encoded into a
        qubit.

        Args:
            x (np.ndarray of shape (n_features,)): Input vector.

        Returns:
            np.ndarray: Quantum state described with an amplitude vector.

        Raises:
            ValueError: When an invalid input is provided.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(f'Invalid input type provided. Expected '
                             f'np.ndarray, got {type(x)} instead.')
        elif len(x.shape) == 1:
            return self._encoding_single(x)
        else:
            raise ValueError(f'Invalid input shape provided. Expected 1D or 2D'
                             f', got {x.shape} instead.')

    def _encoding_single(self, x: np.ndarray) -> np.ndarray:
        """Application of angle encoding to a single sample.

        Args:
            x (np.ndarray of shape(n_features,)): Input sample.

        Returns:
            np.ndarray: Quantum state described with an amplitude vector.
        """
        # Encoding of each feature into a qubit and use of Kronecker product to
        # build the amplitude vector.
        state = 1
        for i in range(x.shape[0]):
            qubit = np.array([np.cos(x[i]), np.sin(x[i])])
            state = np.kron(state, qubit)

        return state
