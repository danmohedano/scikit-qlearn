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

    for i in range(x.shape[0]):
        # Addition of every state to create superposition
        state += _basis_encoding_single(x[i], size)

    # Normalization of the state
    state /= np.sqrt(x.shape[0])

    return state


def amplitude_encoding(x: np.ndarray) -> np.ndarray:
    """Application of amplitude encoding to the input.

    Args:
        x (np.ndarray of shape (n_features,) or (n_samples, n_features)):
            Input. This can be a single sample of shape (n_features,) or a
            dataset of shape (n_samples, n_features). The input MUST be
            normalized. If it is not, use amplitude_encoding_expanded()
            instead.

    Returns:
        np.ndarray: Quantum state described as an amplitude vector. If a
            dataset is provided, the states are concatenated.

    Raises:
        ValueError: If an invalid input is provided.
    """
    if len(x.shape) == 1:
        return _amplitude_encoding_single(x)
    elif len(x.shape) == 2:
        return _amplitude_encoding_dataset(x)
    else:
        raise ValueError(f'Invalid input shape provided. Expected 1D or 2D, '
                         f'got {x.shape} instead.')


def _amplitude_encoding_single(x: np.ndarray) -> np.ndarray:
    """Application of amplitude encoding to a single sample.

    Args:
        x (np.ndarray of shape (n_features,)): Input sample.

    Returns:
        np.ndarray: Quantum state described as an amplitude vector.
    """
    size = int(np.ceil(np.log2(x.shape[0]))**2)

    # Return array with padded 0s at the end
    return np.pad(x, (0, size - x.shape[0]))


def _amplitude_encoding_dataset(x: np.ndarray) -> np.ndarray:
    """Application of amplitude encoding to a dataset.

    Args:
        x (np.ndarray of shape (n_samples, n_features)): Input dataset.

    Returns:
        np.ndarray: Quantum state described as an amplitude vector constructed
            by concatenating the quantum states for each sample.
    """
    states = []
    for i in range(x.shape[0]):
        states += _amplitude_encoding_single(x[i, :]).tolist()

    return np.array(states)


def amplitude_encoding_expanded(x: np.ndarray) -> np.ndarray:
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
        return _amplitude_encoding_expanded_single(x)
    elif len(x.shape) == 2:
        return _amplitude_encoding_expanded_dataset(x)
    else:
        raise ValueError(f'Invalid input shape provided. Expected 1D or 2D, '
                         f'got {x.shape} instead.')


def _amplitude_encoding_expanded_single(x: np.ndarray) -> np.ndarray:
    """Application of expanded amplitude encoding to a single sample.

    Args:
        x (np.ndarray of shape (n_features,)): Input sample.

    Returns:
        np.ndarray: Quantum state described as an amplitude vector.
    """
    # Encode the vector with an extra feature of value 1.0
    state = _amplitude_encoding_single(np.pad(x, (0, 1), constant_values=1.0))

    # Normalize the vector to make it a viable quantum state
    return state / np.linalg.norm(state)


def _amplitude_encoding_expanded_dataset(x: np.ndarray) -> np.ndarray:
    """Application of amplitude encoding to a dataset.

    Args:
        x (np.ndarray of shape (n_samples, n_features)): Input dataset.

    Returns:
        np.ndarray: Quantum state described as an amplitude vector constructed
            by concatenating the quantum states for each sample.
    """
    states = []
    for i in range(x.shape[0]):
        states += _amplitude_encoding_expanded_single(x[i, :]).tolist()

    return np.array(states)


def angle_encoding(x: np.ndarray) -> np.ndarray:
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
        return _angle_encoding_single(x)
    else:
        raise ValueError(f'Invalid input shape provided. Expected 1D or 2D, '
                         f'got {x.shape} instead.')


def _angle_encoding_single(x: np.ndarray) -> np.ndarray:
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


def qsample_encoding(x: np.ndarray) -> np.ndarray:
    """Application of qsample encoding to the input distribution.

    Args:
        x (np.ndarray of shape(n_probs)): Input probability distribution.

    Returns:
        np.ndarray: Quantum state described with an amplitude vector.

    Raises:
        ValueError: When an invalid input is provided.
    """
    if len(x.shape) == 1:
        return _angle_encoding_single(x)
    else:
        raise ValueError(f'Invalid input shape provided. Expected 1D or 2D, '
                         f'got {x.shape} instead.')


def _qsample_encoding_single(x: np.ndarray) -> np.ndarray:
    """Application of qsample encoding to the input distribution.

    Args:
        x (np.ndarray of shape(n_probs)): Input probability distribution.

    Returns:
        np.ndarray: Quantum state described with an amplitude vector.
    """
    size = int(np.ceil(np.log2(x.shape[0]))**2)
    state = np.zeros(size)

    for i in range(x.shape[0]):
        state[i] = np.sqrt(x[i])

    return state
