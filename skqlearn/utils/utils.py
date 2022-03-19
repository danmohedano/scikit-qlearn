import numpy as np
from qiskit import *
from qiskit.providers.aer import QasmSimulator


def fidelity_estimation(
        state_a: np.ndarray,
        state_b: np.ndarray,
) -> float:
    """Fidelity estimation between two quantum states.

    Args:
        state_a: State a described by its amplitudes.
        state_b: State b described by its amplitudes.

    Returns:
        float: Estimation of the fidelity between the states.
    """
    # Calculation of the amount of qubits needed to represent each state
    qubit_size_a = np.ceil(np.log2(state_a.shape[0]))
    qubit_size_b = np.ceil(np.log2(state_b.shape[0]))

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

    # Addition of CSWAP gate with ancilla qubit as control qubit
    for i in range(1, min(qubit_size_a.size, qubit_size_b.size) + 1):
        circuit.cswap(ancilla_qubit,
                      quantum_register_a[-i],
                      quantum_register_b[-i])

    # Addition of Hadamard Gate to ancilla qubit
    circuit.h(ancilla_qubit)

    # Addition of measurement from ancilla qubit to classical bit register
    circuit.measure(ancilla_qubit, classical_register)

    # Simulation
    simulator = QasmSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=10000)
    result = job.result()

    return 2.0 * result.get_counts(compiled_circuit)['0'] / 10000 - 1.0


def distance_estimation(
        a: np.ndarray,
        a_norm: float,
        b: np.ndarray,
        b_norm: float,
) -> float:
    """Euclidean distance estimation through fidelity estimation.

    Args:
        a: Input a.
        a_norm: L2-norm of input a.
        b: Input b.
        b_norm: L2-norm of input b.

    Returns:
        float: Square of the euclidean distance estimated.
    """
    if a.shape != b.shape:
        raise ValueError(f'Vector dimensions disparity between {a.shape} '
                         'and {b.shape}')

    z = a_norm ** 2 + b_norm ** 2
    phi = np.array([a_norm, -b_norm]) / np.sqrt(z)
    psi = np.concatenate([a / a_norm, b / b_norm]) / np.sqrt(2)

    fidelity = fidelity_estimation(phi, psi)
    return 2.0 * z * fidelity

