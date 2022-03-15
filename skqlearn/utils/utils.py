import numpy as np
from qiskit import *
from qiskit.providers.aer import QasmSimulator


def fidelity_estimation(a: np.ndarray, b: np.ndarray, n_iters: int = 1024) -> float:
    """
    Fidelity estimation between two states based on the Control-SWAP Test.
    :param a: State 1 of operation.
    :param b: State 2 of operation.
    :param n_iters: Number of iterations.
    :return: Fidelity of the states.
    """
    size_a = np.ceil(np.log2(a.shape[0]))
    size_b = np.ceil(np.log2(b.shape[0]))

    # Creation of quantum circuit
    a_qr = QuantumRegister(size_a, 'a')
    b_qr = QuantumRegister(size_b, 'b')
    anc = QuantumRegister(1, 'anc')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(a_qr, b_qr, anc, cr)

    # Initialization
    qc.initialize(a, a_qr)
    qc.initialize(b, b_qr)

    # Addition of quantum logic gates
    qc.h(anc)

    for i in range(1, min(a_qr.size, b_qr.size) + 1):
        qc.cswap(anc, a_qr[-i], b_qr[-i])

    qc.h(anc)
    qc.measure(anc, cr)

    #print(qc)

    # Simulation
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=n_iters)

    result = job.result()
    #print(result.get_counts(compiled_circuit))

    return 2 * result.get_counts(compiled_circuit)['0'] / n_iters - 1


def distance_estimation(a: np.ndarray, a_norm: float, b: np.ndarray, b_norm: float, n_iters: int = 1024) -> float:
    """
    Euclidean distance estimation through the use of fidelity.
    :param a:
    :param a_norm:
    :param b:
    :param b_norm:
    :param n_iters:
    :return:
    """
    z = a_norm ** 2 + b_norm ** 2
    phi = np.array([a_norm, -b_norm]) / np.sqrt(z)
    psi = np.concatenate([a / a_norm, b / b_norm]) / np.sqrt(2)

    fidelity = fidelity_estimation(phi, psi, n_iters)
    return np.sqrt(2 * z * fidelity)

