import numpy as np
from typing import Union


def basis_encoding(x: Union[int, np.ndarray]) -> np.ndarray:
    """Application of basis encoding to the input.

    In basis encoding, each classical bit is mapped into a qubit.

    Args:
        x (int or np.ndarray): Either a single sample (int) or a dataset of
            shape (n_samples,).

    Returns:
        np.ndarray: Amplitude vector describing the input encoded into a
            quantum state. If a dataset is provided, the quantum state will be
            a superposition of the encodings of every sample in the dataset.

    Raises:
        ValueError: When an invalid input type is provided.
    """
    if isinstance(x, int) and x >= 0:
        return _basis_encoding_single(x)
    elif isinstance(x, np.ndarray):
        return _basis_encoding_dataset(x)
    else:
        raise ValueError('Invalid input type. Expected positive integer or '
                         f'np.ndarray, got {type(x)} instead.')


def _basis_encoding_single(x: int, size: int = -1) -> np.ndarray:
    """Application of basis encoding to a single input.

    Args:
        x (int): Input value.
        size (int): Size (in n_amplitudes) of the desired state. Must be a
            power of 2. If none is provided, then the size will be determined
            by the next larger power of 2.

    Returns:
        np.ndarray: Quantum state described as an amplitude vector.
    """
    if size == -1:
        # Calculate the closest larger power of 2 (equivalent to calculating
        # the amount of qubits necessary and the size of the amplitude vector
        # needed to represent that state)
        size = int(np.ceil(np.log2(x+1))**2)

    state = np.zeros(size)
    state[x] = 1.0

    return state


def _basis_encoding_dataset(x: np.ndarray) -> np.ndarray:
    """Application of basis encoding to a dataset.

    Args:
        x (np.ndarray of shape (n_samples,)): Input dataset.

    Returns:
        np.ndarray: Quantum state described as an amplitude vector representing
            a superposition of all states in the dataset.
    """
    max_data = np.amax(x)
    size = int(np.ceil(np.log2(max_data+1))**2)
    state = np.zeros(size)

    for data in x:
        # Addition of every state to create superposition
        state += _basis_encoding_single(data, size)

    # Normalization of the state
    state /= np.sqrt(x.shape[0])

    return state

