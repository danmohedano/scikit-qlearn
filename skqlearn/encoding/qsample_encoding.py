from .base_encoding import Encoding
import numpy as np


class QSampleEncoding(Encoding):
    r"""QSample encoding method. :cite:`schuld2018supervised`

    In QSample encoding, a discrete probability distribution is mapped into the
    amplitude vector of a quantum state, defining the encoding feature map as:

    .. math::
       \phi:p(x)\rightarrow \ket{p(x)}=\sum_{X} \sqrt{p(x_i)}\ket{x_i}

    Because the amplitudes are defined as :math:`\alpha_i = \sqrt{p(x_i)}`,
    the resulting quantum state is valid:
    :math:`\sum |\alpha_i|^2=\sum p(x_i) = 1`
    """

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Application of qsample encoding to the input distribution.

        Args:
            x (numpy.ndarray of shape(n_probs)): Input probability distribution.

        Returns:
            numpy.ndarray:
                Quantum state described with an amplitude vector.

        Raises:
            ValueError: When an invalid input is provided.

        Examples:
            >>> a = np.array([0.5, 0.5])
            >>> QSampleEncoding().encoding(a)
            array([0.70710678, 0.70710678])
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
            x (numpy.ndarray of shape(n_probs)): Input probability distribution.

        Returns:
            numpy.ndarray:
                Quantum state described with an amplitude vector.
        """
        size = max(int(2 ** np.ceil(np.log2(x.shape[0]))), 2)
        state = np.zeros(size)

        for i in range(x.shape[0]):
            state[i] = np.sqrt(x[i])

        return state
