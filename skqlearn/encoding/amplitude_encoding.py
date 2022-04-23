from .encoding import Encoding
from typing import Union
import numpy as np


class AmplitudeEncoding(Encoding):
    """Amplitude encoding method.
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to the input.

        Args:
            x (np.ndarray of shape (n_features,) or (n_samples, n_features)):
                Input. This can be a single sample of shape (n_features,) or a
                dataset of shape (n_samples, n_features). The input MUST be
                normalized.

        Returns:
            np.ndarray: Quantum state described as an amplitude vector. If a
                dataset is provided, the states are concatenated.

        Raises:
            ValueError: If an invalid input is provided.
        """
        if len(x.shape) == 1:
            return self._encoding_single(x)
        elif len(x.shape) == 2:
            return self._encoding_dataset(x)
        else:
            raise ValueError(f'Invalid input shape provided. Expected 1D or 2D, '
                             f'got {x.shape} instead.')

    def _encoding_single(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to a single sample.

        Args:
            x (np.ndarray of shape (n_features,)): Input sample.

        Returns:
            np.ndarray: Quantum state described as an amplitude vector.
        """
        size = int(np.ceil(np.log2(x.shape[0])) ** 2)

        # Return array with padded 0s at the end
        return np.pad(x, (0, size - x.shape[0]))

    def _encoding_dataset(self, x: np.ndarray) -> np.ndarray:
        """Application of amplitude encoding to a dataset.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Input dataset.

        Returns:
            np.ndarray: Quantum state described as an amplitude vector constructed
                by concatenating the quantum states for each sample.
        """
        vector_size = int(np.ceil(np.log2(x.shape[1])) ** 2)
        states = np.zeros(vector_size * x.shape[0])
        for i in range(x.shape[0]):
            states[i * vector_size:(i + 1) * vector_size] = \
                self._encoding_single(x[i, :])

        return np.array(states)
