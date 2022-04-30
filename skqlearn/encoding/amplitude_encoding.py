from .encoding import Encoding
import numpy as np


class AmplitudeEncoding(Encoding):
    r"""Amplitude encoding method. :cite:`schuld2018supervised`

    In amplitude encoding, each component of the input vector
    :math:`\boldsymbol{x} \in \mathbb{R}^N` is mapped to an amplitude of the
    quantum state, defining the encoding feature map as:

    .. math::
       \phi:\boldsymbol{x}\rightarrow\ket{\psi_\boldsymbol{x}}=\sum_{i=1}^{N}
       x_i\ket{i}

    In order to represent a valid quantum state the amount of amplitudes, and
    therefore, the dimension of the vectors must be a power of 2,
    :math:`N=2^n`. If they are not, they will be padded with zeros at the end.

    Therefore, the kernel defined by the inner product is the linear kernel:

    .. math::
       k(\boldsymbol{x}, \boldsymbol{x'}) = \braket{\psi_{\boldsymbol{x}}|
       \psi_{\boldsymbol{x'}}} = \boldsymbol{x}^T\boldsymbol{x'}

    A dataset can be encoded by concatenating all the input vectors.
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to the input.

        Args:
            x (np.ndarray of shape (n_features,) or (n_samples, n_features)):
                Input. This can be a single sample of shape (n_features,) or a
                dataset of shape (n_samples, n_features).

                .. note::
                   The input must be normalized in order to define a valid
                   quantum state. Refer to `ExpandedAmplitudeEncoding` if the
                   data is not normalized.
        Returns:
            np.ndarray:
                Quantum state described as an amplitude vector. If a
                dataset is provided, the states are concatenated.

        Raises:
            ValueError: If an invalid input type is provided or it is not
                normalized.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(f'Invalid input type provided. Expected '
                             f'np.ndarray got {type(x)} instead.')
        elif len(x.shape) == 1:
            return self._encoding_single(x)
        elif len(x.shape) == 2:
            return self._encoding_dataset(x)
        else:
            raise ValueError(f'Invalid input shape provided. Expected 1D or 2D'
                             f', got {x.shape} instead.')

    def _encoding_single(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to a single sample.

        Args:
            x (np.ndarray of shape (n_features,)): Input sample.

        Returns:
            np.ndarray: Quantum state described as an amplitude vector.

        Raises:
            ValueError: If the input is not normalized.
        """
        if np.linalg.norm(x) != 1.0:
            raise ValueError(f'Invalid input, must be normalized. Got |x| = '
                             f'{np.linalg.norm(x)} instead')

        size = max(int(2 ** np.ceil(np.log2(x.shape[0]))), 2)

        # Return array with padded 0s at the end
        return np.pad(x, (0, size - x.shape[0]))

    def _encoding_dataset(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to a dataset.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Input dataset.

        Returns:
            np.ndarray: Quantum state described as an amplitude vector
                constructed by concatenating the quantum states for each
                sample.
        """
        vector_size = max(int(2 ** np.ceil(np.log2(x.shape[1]))), 2)
        states = np.zeros(vector_size * x.shape[0])
        for i in range(x.shape[0]):
            states[i * vector_size:(i + 1) * vector_size] = \
                self._encoding_single(x[i, :])

        return np.array(states)
