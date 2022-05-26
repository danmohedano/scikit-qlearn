from .amplitude_encoding import AmplitudeEncoding
from .base_encoding import Encoding
from skqlearn.utils import inner_product_estimation
import numpy as np


class ExpandedAmplitudeEncoding(Encoding):
    r"""Expanded amplitude encoding method. :cite:`schuld2018supervised`

    This encoding method tries to solve the normalization problem in regular
    Amplitude Encoding. If non-normalized data is normalized for use on
    Amplitude Encoding, the data will lose one dimension of information. For
    example, if a 2D point is normalized, it will be mapped into the unit
    circle, a 1D shape. By adding an extra component to
    :math:`\boldsymbol{x}\in\mathbb{R}^N` with a value of :math:`c`,
    :math:`x_{0}=1`, and then normalizing, the information loss is mitigated.

    The encoding and produced kernel are work identical to regular Amplitude
    Encoding's.

    .. math::
       \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
       \frac{1}{|\boldsymbol{x}|^2+c^2}\left(c\left|0\right> +
       \sum_{i=1}^{N}x_i\left|i\right>\right)

    Attributes:
        degree (int): Desired degree of the polynomial kernel defined by the
            encoding. In turn, it defines the amount of copies of each input
            vector that are encoded into the quantum state.
        c (float): Constant to expand the input vectors with.
    """
    def __init__(self, degree: int = 1, c: float = 1.0):
        """Construct a ExpandedAmplitudeEncoding object.

        Args:
            degree (int): Degree of the encoding (number of copies of the
                states that will be created).
            c (float): Constant to expand the input vectors with.

        Raises:
            ValueError: if the degree is smaller than 1.
        """
        if degree < 1:
            raise ValueError(f'Invalid degree provided. Got {degree} instead.')
        self.degree = degree
        self.c = c

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of expanded amplitude encoding to the input.

        Expanded amplitude encoding allows for non-normalized vector to be
        encoded as quantum states. This is achieved by the inclusion of an
        extra feature/component with value :math:`c`.

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
            >>> ExpandedAmplitudeEncoding(c=1.0).encoding(a)
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
        normalized_x = np.pad(x.astype(float), (1, 0), constant_values=self.c)
        normalized_x /= np.linalg.norm(normalized_x)
        state = AmplitudeEncoding(self.degree).encoding(normalized_x)

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
        vector_size = max(int(2 ** np.ceil(np.log2(x.shape[1] + 1))), 2)
        vector_size = vector_size ** self.degree

        states = np.zeros(vector_size * x.shape[0])
        for i in range(x.shape[0]):
            states[i * vector_size:(i + 1) * vector_size] = \
                self._encoding_single(x[i, :])

        # Normalization of the concatenated vectors
        states /= np.sqrt(x.shape[0])

        # Extend dimensions for correct quantum state definition (power of 2)
        states_size = max(int(2 ** np.ceil(np.log2(states.shape[0]))), 2)
        states = np.pad(states.astype(float),
                        (0, states_size - states.shape[0]),
                        constant_values=0.0)

        return states

    def classic_kernel(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> np.ndarray:
        """Classical calculation of the kernel formed by the encoding and the
        inner product.

        Args:
            x (numpy.ndarray of shape (n_samples_1, n_features)): First input.
            y (numpy.ndarray of shape (n_samples_2, n_features)): Second input.

        Returns:
            numpy.ndarray of shape (n_samples_1, n_samples_2):
                Resulting kernel matrix.
        """
        # Compute norms of input vectors
        x_norms = [np.linalg.norm(x[i, :]) for i in range(x.shape[0])]
        y_norms = [np.linalg.norm(y[i, :]) for i in range(y.shape[0])]

        # Application of the encoding to the inputs
        x_samples_list = [self.encoding(x[i, :]) for i in range(x.shape[0])]
        y_samples_list = [self.encoding(y[i, :]) for i in range(y.shape[0])]

        # Pad with zeros if dimensions differ
        x_size = max([x.shape[0] for x in x_samples_list])
        y_size = max([y.shape[0] for y in y_samples_list])
        x_samples_list = [np.pad(x, (0, x_size - x.shape[0]))
                          for x in x_samples_list]
        y_samples_list = [np.pad(y, (0, y_size - y.shape[0]))
                          for y in y_samples_list]

        x_encoded = np.vstack(x_samples_list)
        y_encoded = np.vstack(y_samples_list)

        # Calculate gram matrix and correct for the normalization
        gram = np.dot(x_encoded, y_encoded.T)

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                factor_x = np.sqrt(x_norms[i] ** 2 + self.c ** 2)
                factor_y = np.sqrt(y_norms[j] ** 2 + self.c ** 2)
                gram[i, j] *= (factor_x * factor_y) ** self.degree

        return gram

    def quantum_kernel(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> np.ndarray:
        """Quantum estimation of the kernel formed by the encoding and the
        inner product.

        Args:
            x (numpy.ndarray of shape (n_samples_1, n_features)): First input.
            y (numpy.ndarray of shape (n_samples_2, n_features)): Second input.

        Returns:
            numpy.ndarray of shape (n_samples_1, n_samples_2):
                Resulting kernel matrix.
        """
        # Compute norms of input vectors
        x_norms = [np.linalg.norm(x[i, :]) for i in range(x.shape[0])]
        y_norms = [np.linalg.norm(y[i, :]) for i in range(y.shape[0])]

        # Application of the encoding to the inputs
        x_samples_list = [self.encoding(x[i, :]) for i in range(x.shape[0])]
        y_samples_list = [self.encoding(y[i, :]) for i in range(y.shape[0])]

        # Pad with zeros if dimensions differ
        x_size = max([x.shape[0] for x in x_samples_list])
        y_size = max([y.shape[0] for y in y_samples_list])
        x_samples_list = [np.pad(x, (0, x_size - x.shape[0]))
                          for x in x_samples_list]
        y_samples_list = [np.pad(y, (0, y_size - y.shape[0]))
                          for y in y_samples_list]

        x_encoded = np.vstack(x_samples_list)
        y_encoded = np.vstack(y_samples_list)

        # Calculation of the gram matrix
        gram = np.zeros([x.shape[0], y.shape[0]])
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                factor_x = np.sqrt(x_norms[i] ** 2 + self.c ** 2)
                factor_y = np.sqrt(y_norms[j] ** 2 + self.c ** 2)
                factor = (factor_x * factor_y) ** self.degree
                gram[i, j] = factor * inner_product_estimation(x_encoded[i, :],
                                                               y_encoded[j, :])

        return gram
