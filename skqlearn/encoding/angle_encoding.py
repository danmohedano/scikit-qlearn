from .encoding import Encoding
from typing import Union
import numpy as np


class AngleEncoding(Encoding):
    """Angle encoding method.
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of angle encoding to the input.

        In angle encoding, each feature of the input vector is encoded into a
        qubit.

        todo: Explanation

        Args:
            x (np.ndarray of shape (n_features,)): Input vector.

        Returns:
            np.ndarray: Quantum state described with an amplitude vector.

        Raises:
            ValueError: When an invalid input is provided.
        """
        if len(x.shape) == 1:
            return self._encoding_single(x)
        else:
            raise ValueError(f'Invalid input shape provided. Expected 1D or 2D, '
                             f'got {x.shape} instead.')

    def _encoding_single(self, x: np.ndarray) -> np.ndarray:
        """Application of angle encoding to a single sample.

        Args:
            x (np.ndarray of shape(n_features,)): Input sample.

        Returns:
            np.ndarray: Quantum state described with an amplitude vector.
        """
        # Encoding of each feature into a qubit and use of Kronecker product to
        # build the amplitude vector.
        state = 1
        for i in range(x.shape[0]):
            qubit = np.array([np.cos(x[i]), np.sin(x[i])])
            state = np.kron(state, qubit)

        return state
