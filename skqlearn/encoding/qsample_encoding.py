from .encoding import Encoding
import numpy as np


class QSampleEncoding(Encoding):
    """QSample encoding method.
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of qsample encoding to the input distribution.

        Args:
            x (np.ndarray of shape(n_probs)): Input probability distribution.

        Returns:
            np.ndarray: Quantum state described with an amplitude vector.

        Raises:
            ValueError: When an invalid input is provided.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(f'Invalid input type provided. Expected '
                             f'np.ndarray got {type(x)} instead.')
        elif len(x.shape) == 1:
            return self._encoding_single(x)
        else:
            raise ValueError(f'Invalid input shape provided. Expected 1D or 2D'
                             f', got {x.shape} instead.')

    def _encoding_single(self, x: np.ndarray) -> np.ndarray:
        """Application of qsample encoding to the input distribution.

        Args:
            x (np.ndarray of shape(n_probs)): Input probability distribution.

        Returns:
            np.ndarray: Quantum state described with an amplitude vector.
        """
        size = max(int(2 ** np.ceil(np.log2(x.shape[0]))), 2)
        state = np.zeros(size)

        for i in range(x.shape[0]):
            state[i] = np.sqrt(x[i])

        return state
