import qiskit
from qiskit import QuantumRegister, QuantumCircuit


def multiqubit_cswap(
        qubit_size_a: int,
        qubit_size_b: int,
        name: str = "MQ-CSWAP",
) -> qiskit.circuit.Instruction:
    """Create a multi-qubit controlled SWAP quantum gate.

    Composes a multi-qubit controlled SWAP gate as a quantum circuit by
    applying CSWAP gates, qubit-wise, to registers a and b, using as control
    an ancilla qubit. If the dimensions of a and b differ, then the CSWAP's are
    applied until there are no more qubits in one of the states.

    Args:
        qubit_size_a (int): Size in qubits for register a.
        qubit_size_b (int): Size in qubits for register b.
        name (str): Name for the circuit.

    Returns:
        qiskit.circuit.Instruction: The gate composed as a quantum circuit. It
         can then be appended in other quantum circuits.

    Raises:
        ValueError: If negative or zero sizes are provided.
    """
    if qubit_size_a < 1 or qubit_size_b < 1:
        raise ValueError('Invalid qubit sizes. Must be > 0.')
    # Creation of the quantum registers that will store states a and b
    quantum_register_a = QuantumRegister(qubit_size_a, 'a')
    quantum_register_b = QuantumRegister(qubit_size_b, 'b')

    # Creation of the ancilla qubit used as control in the CSWAP gate
    ancilla_qubit = QuantumRegister(1, 'anc')

    # Creation of the quantum circuit where the gate is being composed
    circuit = QuantumCircuit(quantum_register_a, quantum_register_b,
                             ancilla_qubit, name=name)

    # Addition of CSWAP gates with ancilla qubit as control qubit
    for i in range(1, min(qubit_size_a, qubit_size_b) + 1):
        circuit.cswap(ancilla_qubit,
                      quantum_register_a[-i],
                      quantum_register_b[-i])

    return circuit.to_instruction()
