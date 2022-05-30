from .amplitude_encoding import AmplitudeEncoding
from .base_encoding import Encoding
import numpy as np
from typing import Tuple


class ExpandedAmplitudeEncoding(Encoding):
    r"""Expanded amplitude encoding method. :cite:`schuld2018supervised`

    This encoding method tries to solve the normalization problem in regular
    Amplitude Encoding. If non-normalized data is normalized for use on
    Amplitude Encoding, the data will lose one dimension of information. For
    example, if a 2D point is normalized, it will be mapped into the unit
    circle, a 1D shape. By adding an extra component to
    :math:`\boldsymbol{x}\in\mathbb{R}^N` with a value of :math:`c`,
    :math:`x_{0}=c`, and then normalizing, the information loss is mitigated.

    The encoding and produced kernel work identical to regular Amplitude
    Encoding's.

    .. math::
       \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
       \frac{1}{\sqrt{|\boldsymbol{x}|^2+c^2}}\left(c\left|0\right> +
       \sum_{i=1}^{N}x_i\left|i\right>\right)

    .. math::
        k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
        \psi_{\boldsymbol{x'}}\right> = \frac{1}{\sqrt{|\boldsymbol{x}|^2+c^2}
        \sqrt{|\boldsymbol{x'}|^2+c^2}}\boldsymbol{x}^T\boldsymbol{x'}

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

    def _sample_preparation(
            self,
            x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample preparation for the computation of kernels.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Input samples.

        Returns:
            numpy.ndarray of shape (n_samples, n_encoded_features):
                Encoded samples.
            numpy.ndarray of shape (n_samples,):
                Calculated norms for the samples previous to encoding. Might
                be necessary for the correction of the inner product.
        """
        # Obtain the norms of the input samples
        x_norms = np.linalg.norm(x, axis=1)

        # Compute amount of features after encoding. The amount of features
        # will be of the form 2^n to define a valid quantum state. This is done
        # accounting for the extra component introduced.
        n_encoded_features = max(int(2 ** np.ceil(np.log2(x.shape[1] + 1))), 2)

        # Pad samples with zeros and include extra component with value c
        x_encoded = np.zeros((x.shape[0], n_encoded_features))
        x_encoded[:, 0] += self.c
        x_encoded[:, 1:x.shape[1]+1] = x

        # Normalize new expanded samples
        x_encoded /= np.sqrt(x_norms ** 2 + self.c ** 2)[:, None]

        if self.degree > 1:
            # The kronecker product is applied row-wise to every sample
            def kron_gen(i):
                sample = x_encoded[i, :]
                for _ in range(self.degree - 1):
                    sample = np.kron(sample, x_encoded[i, :])

                return sample

            x_encoded = np.array([kron_gen(i) for i in range(x.shape[0])])

        return x_encoded, x_norms

    @property
    def _is_correction_needed(self) -> bool:
        """Indicates if the kernel calculation needs a correction or not.

        Returns:
            True
        """
        return True

    def _correction_factor(
            self,
            x_norm: float,
            y_norm: float
    ) -> float:
        """Correction factor for a kernel between x and y.

        Args:
            x_norm (float): Norm of first input.
            y_norm (float): Norm of second input.

        Returns:
            float:
                Correction factor for the specific kernel.
        """
        x_correction = x_norm ** 2 + self.c ** 2
        y_correction = y_norm ** 2 + self.c ** 2
        return np.sqrt(x_correction * y_correction) ** self.degree
