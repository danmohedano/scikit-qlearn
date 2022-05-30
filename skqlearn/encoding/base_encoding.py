from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np
from skqlearn.utils import inner_product_estimation


class Encoding(ABC):
    r"""Abstract class to define a data encoding method.

    Each implementation must define the `encoding` method as a feature map
    :math:`\phi : \mathcal{X} \rightarrow \mathcal{F}` taking a sample
    :math:`x` from the input space :math:`\mathcal{X}` to the feature space
    :math:`\mathcal{F}`.
    """

    @abstractmethod
    def encoding(self, x: Union[int, np.ndarray]) -> np.ndarray:
        pass

    def _sample_preparation(
            self,
            x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample preparation for the computation of kernels.

        Encodes every sample of the input. Can be overriden to increase
        efficiency for each specific encoding method.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Input samples.

        Returns:
            numpy.ndarray of shape (n_samples, n_encoded_features):
                Encoded samples.
            numpy.ndarray of shape (n_samples,):
                Calculated norms for the samples previous to encoding. Might
                be necessary for the correction of the inner product.
        """
        # Application of the encoding to the inputs
        x_norms = np.linalg.norm(x, axis=1)
        x_samples_list = [self.encoding(x[i, :]) for i in range(x.shape[0])]

        # Pad with zeros if dimensions differ
        x_size = max([x.shape[0] for x in x_samples_list])
        x_encoded = np.zeros((x.shape[0], x_size))
        for i in range(len(x_samples_list)):
            sample_i = x_samples_list[i]
            x_encoded[i, :sample_i.shape[0]] = sample_i

        return x_encoded, x_norms

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
        return 1.0

    @property
    def _is_correction_needed(self) -> bool:
        """Indicates if the kernel calculation needs a correction or not.

        Avoids unnecessary calculations.

        Returns:
            bool:
                True if a correction is needed, False otherwise
        """
        return False

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
        # Encoding of the input samples
        x_encoded, x_norms = self._sample_preparation(x)
        y_encoded, y_norms = self._sample_preparation(y)

        # Calculation of the Gram matrix
        gram = np.dot(x_encoded, y_encoded.T)

        # Correction of the calculated Gram matrix if necessary
        if self._is_correction_needed:
            correction_matrix = np.array([[self._correction_factor(x_norms[i],
                                                                   y_norms[j])
                                           for j in range(gram.shape[1])]
                                          for i in range(gram.shape[0])])

            gram *= correction_matrix

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
        # Encoding of the input samples that results in matrices of shape
        # (n_samples, n_features_encoded)
        x_encoded, x_norms = self._sample_preparation(x)
        y_encoded, y_norms = self._sample_preparation(y)

        # Calculation of the gram matrix
        gram = np.zeros((x.shape[0], y.shape[0]))

        for i in range(gram.shape[0]):
            for j in range(gram.shape[1]):
                gram[i, j] = inner_product_estimation(x_encoded[i, :],
                                                      y_encoded[j, :])
                if self._is_correction_needed:
                    gram[i, j] *= self._correction_factor(x_norms[i],
                                                          y_norms[j])

        return gram
