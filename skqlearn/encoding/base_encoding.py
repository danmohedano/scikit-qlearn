from abc import ABC, abstractmethod
from typing import Union
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
        # Application of the encoding to the inputs
        x_samples_list = [self.encoding(x[i, :]) for i in range(x.shape[0])]
        y_samples_list = [self.encoding(y[i, :]) for i in range(y.shape[0])]

        x_encoded = np.vstack(x_samples_list)
        y_encoded = np.vstack(y_samples_list)

        return np.dot(x_encoded, y_encoded.T)

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
        # Application of the encoding to the inputs
        x_samples_list = [self.encoding(x[i, :]) for i in range(x.shape[0])]
        y_samples_list = [self.encoding(y[i, :]) for i in range(y.shape[0])]

        x_encoded = np.vstack(x_samples_list)
        y_encoded = np.vstack(y_samples_list)

        # Calculation of the gram matrix
        # Because the inputs are expected to be real, the Gram matrix will be
        # symmetric, therefore half of the matrix does not have to be computed
        gram = np.zeros([x.shape[0], y.shape[0]])
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                gram[i, j] = inner_product_estimation(x_encoded[i, :],
                                                      y_encoded[j, :])

        return gram
