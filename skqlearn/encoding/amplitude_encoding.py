from .base_encoding import Encoding
import numpy as np
import math


class AmplitudeEncoding(Encoding):
    r"""Amplitude encoding method. :cite:`schuld2018supervised`

    In amplitude encoding, each component of the input vector
    :math:`\boldsymbol{x} \in \mathbb{R}^N` is mapped to an amplitude of the
    quantum state, defining the encoding feature map as:

    .. math::
       \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
       \sum_{i=1}^{N}x_i\left|i\right>

    In order to represent a valid quantum state the amount of amplitudes, and
    therefore, the dimension of the vectors must be a power of 2,
    :math:`N=2^n`. If they are not, they will be padded with zeros at the end.

    Therefore, the kernel defined by the inner product is the linear kernel:

    .. math::
       k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
       \psi_{\boldsymbol{x'}}\right> = \boldsymbol{x}^T\boldsymbol{x'}

    By, instead, mapping the input to :math:`d` copies of an amplitude
    encoded quantum state, a polynomial kernel can be defined:

    .. math::
       \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>
       ^{\bigotimes d}

    .. math::
       k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
       \psi_{\boldsymbol{x'}}\right> \bigotimes ... \bigotimes
       \left<\psi_{\boldsymbol{x}}|\psi_{\boldsymbol{x'}}\right> =
       (\boldsymbol{x}^T\boldsymbol{x'})^d

    A dataset can be encoded by concatenating all the input vectors.

    Attributes:
        degree (int): Desired degree of the polynomial kernel defined by the
            encoding. In turn, it defines the amount of copies of each input
            vector that are encoded into the quantum state.
    """
    def __init__(self, degree: int = 1):
        """Construct a AmplitudeEncoding object.

        Args:
            degree (int): Degree of the encoding (number of copies of the
                states that will be created).

        Raises:
            ValueError: if the degree is smaller than 1.
        """
        if degree < 1:
            raise ValueError(f'Invalid degree provided. Got {degree} instead.')
        self.degree = degree

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to the input.

        Args:
            x (numpy.ndarray of shape (n_features) or (n_samples, n_features)):
                Input. This can be a single sample of shape (n_features,) or a
                dataset of shape (n_samples, n_features).

                .. note::
                   The input must be normalized in order to define a valid
                   quantum state. Refer to `ExpandedAmplitudeEncoding` if the
                   data is not normalized.
        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector with the amount
                of copies of the state indicated in the constructor. If a
                dataset is provided, the states are concatenated.

        Raises:
            ValueError: If an invalid input type is provided or it is not
                normalized.

        Examples:
            >>> a = np.array([0.5, 0.5, 0.5, 0.5])
            >>> AmplitudeEncoding().encoding(a)
            array([0.5, 0.5, 0.5, 0.5])

            >>> a = np.array([0.0, 1.0, 0.0])
            >>> AmplitudeEncoding().encoding(a)
            array([0., 1., 0., 0.])

            >>> a = np.array([0.0, 1.0, 0.2, 0.0])
            >>> AmplitudeEncoding().encoding(a)
            Traceback (most recent call last):
             ...
            ValueError: Invalid input, must be normalized. Got |x| = 1.019803902718557 instead

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
            x (numpy.ndarray of shape (n_features,)): Input sample.

        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector.

        Raises:
            ValueError: If the input is not normalized.
        """
        if not math.isclose(np.linalg.norm(x), 1.0, abs_tol=1e-8):
            raise ValueError(f'Invalid input, must be normalized. Got |x| = '
                             f'{np.linalg.norm(x)} instead')

        size = max(int(2 ** np.ceil(np.log2(x.shape[0]))), 2)

        # Pad 0s at the end of the state to make it whole qubit-sized
        base_state = np.pad(x, (0, size - x.shape[0]))

        # Create the amount of copies defined by degree
        state = 1
        for i in range(self.degree):
            state = np.kron(state, base_state)

        return state

    def _encoding_dataset(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to a dataset.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Input dataset.

        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector
                constructed by concatenating the quantum states for each
                sample.
        """
        # Calculate the size of a single state, accounting for the degree
        vector_size = max(int(2 ** np.ceil(np.log2(x.shape[1]))), 2)
        vector_size = vector_size ** self.degree

        states = np.zeros(vector_size * x.shape[0])
        for i in range(x.shape[0]):
            states[i * vector_size:(i + 1) * vector_size] = \
                self._encoding_single(x[i, :])

        # Normalization of the concatenated vectors
        states /= np.sqrt(x.shape[0])

        return states
