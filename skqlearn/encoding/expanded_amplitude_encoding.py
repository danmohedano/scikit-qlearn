from .encoding import Encoding
from typing import Union
import numpy as np
from .amplitude_encoding import AmplitudeEncoding


class ExpandedAmplitudeEncoding(Encoding):
    """Expanded Amplitude encoding method.
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of expanded amplitude encoding to the input.

        Expanded amplitude encoding allows for non-normalized vector to be encoded
        as quantum states. This is achieved by the inclusion of an extra feature
        with value 1.

        todo: elaborate on explanation

        Args:
            x (np.ndarray of shape (n_features,) or (n_samples, n_features)):
                Input. This can be a single sample of shape (n_features,) or a
                dataset of shape (n_samples, n_features).

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
        """Application of expanded amplitude encoding to a single sample.

        Args:
            x (np.ndarray of shape (n_features,)): Input sample.

        Returns:
            np.ndarray: Quantum state described as an amplitude vector.
        """
        # Encode the vector with an extra feature of value 1.0
        amp_encoding = AmplitudeEncoding()
        state = amp_encoding.encoding(np.pad(x, (0, 1), constant_values=1.0))

        # Normalize the vector to make it a viable quantum state
        return state / np.linalg.norm(state)

    def _encoding_dataset(self, x: np.ndarray) -> np.ndarray:
        """Application of expanded amplitude encoding to a dataset.

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
