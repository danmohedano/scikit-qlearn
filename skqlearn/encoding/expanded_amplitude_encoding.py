import numpy as np
from .amplitude_encoding import AmplitudeEncoding
from .base_encoding import Encoding


class ExpandedAmplitudeEncoding(Encoding):
    r"""Expanded amplitude encoding method. :cite:`schuld2018supervised`

    This encoding method tries to solve the normalization problem in regular
    Amplitude Encoding. If non-normalized data is normalized for use on
    Amplitude Encoding the data will lose one dimension of information. For
    example, if a 2D point is normalized, it will be mapped into the unit
    circle, a 1D shape. By adding an extra component to
    :math:`\boldsymbol{x}\in\mathbb{R}^N` with a value of :math:`1`,
    :math:`x_{N+1}=1`, and then normalizing, the information loss is avoided.

    The encoding and produced kernel are identical to regular Amplitude
    Encoding's.

    Attributes:
        degree (int): Desired degree of the polynomial kernel defined by the
            encoding. In turn, it defines the amount of copies of each input
            vector that are encoded into the quantum state.
    """
    def __init__(self, degree: int = 1):
        """Construct a ExpandedAmplitudeEncoding object.

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
        """Application of expanded amplitude encoding to the input.

        Expanded amplitude encoding allows for non-normalized vector to be
        encoded as quantum states. This is achieved by the inclusion of an
        extra feature/component with value 1.

        Args:
            x (numpy.ndarray of shape (n_features) or (n_samples, n_features)):
                Input. This can be a single sample of shape (n_features,) or a
                dataset of shape (n_samples, n_features).

        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector. If a
                dataset is provided, the states are concatenated.

        Raises:
            ValueError: If an invalid input is provided.

        Examples:
            >>> a = np.array([1.0, 1.0, 1.0])
            >>> ExpandedAmplitudeEncoding().encoding(a)
            array([0.5, 0.5, 0.5, 0.5])
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
        """Application of expanded amplitude encoding to a single sample.

        Args:
            x (numpy.ndarray of shape (n_features,)): Input sample.

        Returns:
            numpy.ndarray:
                Quantum state described as an amplitude vector.
        """
        # Encode the vector with an extra feature of value 1.0
        normalized_x = np.pad(x.astype(float), (0, 1), constant_values=1.0)
        normalized_x /= np.linalg.norm(normalized_x)
        state = AmplitudeEncoding(self.degree).encoding(normalized_x)

        # Normalize the vector to make it a viable quantum state
        return state

    def _encoding_dataset(self, x: np.ndarray) -> np.ndarray:
        """Application of expanded amplitude encoding to a dataset.

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
