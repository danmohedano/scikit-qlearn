import numpy as np
from qiskit import *
from skqlearn.gates import multiqubit_cswap
from skqlearn.jobhandler import JobHandler


def fidelity_estimation(
        state_a: np.ndarray,
        state_b: np.ndarray,
) -> float:
    """Fidelity estimation between two quantum states.

    Args:
        state_a (np.ndarray): State a described by its amplitudes.
        state_b (np.ndarray): State b described by its amplitudes.

    Returns:
        float: Estimation of the fidelity between the states.
    """
    # Calculation of the amount of qubits needed to represent each state
    qubit_size_a = np.ceil(np.log2(state_a.shape[0])).astype(int)
    qubit_size_b = np.ceil(np.log2(state_b.shape[0])).astype(int)

    # Creation of the quantum registers that will store states a and b, as well
    # as the ancilla qubit needed for the estimation and the classical qubit
    # utilized in the measurement.
    quantum_register_a = QuantumRegister(qubit_size_a, 'a')
    quantum_register_b = QuantumRegister(qubit_size_b, 'b')
    ancilla_qubit = QuantumRegister(1, 'anc')
    classical_register = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(quantum_register_a, quantum_register_b,
                             ancilla_qubit, classical_register)

    # Initialization of the quantum registers with the states provided
    circuit.initialize(state_a, quantum_register_a)
    circuit.initialize(state_b, quantum_register_b)

    # Addition of Hadamard Gate to ancilla qubit
    circuit.h(ancilla_qubit)

    # Addition of the multi-qubit controlled SWAP gate
    circuit.append(multiqubit_cswap(qubit_size_a, qubit_size_b),
                   [*quantum_register_a[:],
                    *quantum_register_b[:],
                    ancilla_qubit])

    # Addition of Hadamard Gate to ancilla qubit
    circuit.h(ancilla_qubit)

    # Addition of measurement from ancilla qubit to classical bit register
    circuit.measure(ancilla_qubit, classical_register)

    # Execution of the circuit
    job_handler = JobHandler()
    result = job_handler.run_job(circuit)

    # Estimation of the probability
    shots = job_handler.shots
    comp = job_handler.compiled_circuits

    return 2.0 * result.get_counts(comp)['0'] / shots - 1.0


def distance_estimation(
        a: np.ndarray,
        a_norm: float,
        b: np.ndarray,
        b_norm: float,
) -> float:
    """Euclidean distance estimation through fidelity estimation.

    todo: Explain negative results problem and solution.

    Args:
        a (np.ndarray): Input a.
        a_norm (float): L2-norm of input a.
        b (np.ndarray): Input b.
        b_norm (float): L2-norm of input b.

    Returns:
        float: Square of the euclidean distance estimated.
    """
    if a.shape != b.shape:
        raise ValueError(f'Vector dimensions disparity between {a.shape} '
                         'and {b.shape}')

    z = a_norm ** 2 + b_norm ** 2
    phi = np.array([a_norm, -b_norm]) / np.sqrt(z)
    psi = np.concatenate([a / a_norm, b / b_norm]) / np.sqrt(2.0)

    fidelity = fidelity_estimation(phi, psi)
    return max(2.0 * z * fidelity, 0.0)


def inner_product_estimation(
        state_a: np.ndarray,
        state_b: np.ndarray,
) -> float:
    """Quantum estimation of the inner product between two quantum states.

    todo: elaborate on algorithm

    Args:
        state_a (np.ndarray): State a described by its amplitudes.
        state_b (np.ndarray): State b described by its amplitudes.

    Returns:
        float: Estimation of the inner product between both quantum states.
    """
    if state_a.shape != state_b.shape:
        # Pad with 0s the amplitude vectors if necessary in order for both
        # states to have the same amount of qubits.
        max_size = max(state_a.shape[0], state_b.shape[0])
        state_a = np.pad(state_a, (0, max_size - state_a.shape[0]))
        state_b = np.pad(state_b, (0, max_size - state_b.shape[0]))

    # Calculation of the amount of qubits needed to represent each state
    qubit_size = np.ceil(np.log2(state_a.shape[0])).astype(int)

    quantum_register = QuantumRegister(qubit_size + 1, 'qr')
    classical_register = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(quantum_register, classical_register)

    # Calculation of the initial quantum state
    psi = np.concatenate([state_a, state_b]) / np.sqrt(2)
    circuit.initialize(psi, quantum_register)

    circuit.h(quantum_register[0])

    # Addition of measurement to classical bit register
    circuit.measure(quantum_register[0], classical_register)

    # Execution of the circuit
    job_handler = JobHandler()
    result = job_handler.run_job(circuit)

    # Estimation of the probability
    shots = job_handler.shots
    comp = job_handler.compiled_circuits

    return 2.0 * result.get_counts(comp)['0'] / shots - 1.0

