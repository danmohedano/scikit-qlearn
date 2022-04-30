import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from skqlearn.gates import multiqubit_cswap
from skqlearn.jobhandler import JobHandler


def fidelity_estimation(
        state_a: np.ndarray,
        state_b: np.ndarray,
) -> float:
    r"""Fidelity estimation between two quantum states by
    SWAP-Test. :cite:`inaquantumworld`

    The fidelity of two quantum states can be defined as:

    .. math::
       Fid(\ket{a}, \ket{b})=|\braket{a|b}|^2

    This is a similarity measure between two quantum states, its value ranging
    from 0 if the states are orthogonal to 1 if they are identical.

    It can be measured by performing a SWAP-Test, with the following
    operations:

    1. Initialize the state with an ancilla qubit and the two n-qubit
    quantum states:

    .. math:: \ket{\psi_0}=\ket{0,a,b}

    2. Apply a Hadamard gate to the control
    qubit:

    .. math::
       \ket{\psi_1} = (H \bigotimes I^{\bigotimes n} \bigotimes
       I^{\bigotimes n})\ket{\psi_0} = \frac{1}{\sqrt{2}}(\ket{0,a,b} +
       \ket{1,a,b})

    3. Apply a CSWAP gate to the three states, using the ancilla qubit as
    control:

    .. math::
       \ket{\psi_2}= \frac{1}{\sqrt{2}}(\ket{0,a,b} + \ket{1,b,a})

    4. Apply a Hadamard gate to the control
    qubit:

    .. math::
       \ket{\psi_3} = \frac{1}{2}\ket{0}(\ket{a,b}+\ket{b,a}) +
       \frac{1}{2} \ket{1}(\ket{a,b}-\ket{b,a})

    5. Measure the probability of the control qubit being
    in state :math:`\ket{0}`:

    .. math::
       P(\ket{0}) &= |\braket{0|\psi_3}|^2 \\
                  &= \frac{1}{4}|(\ket{a,b}+\ket{b,a})|^2 \\
                  &= \frac{1}{2} + \frac{1}{2}|\braket{a|b}|^2

    The algorithm also works for quantum states with different sizes, though
    the notion of similiarity then becomes harder to define.

    Args:
        state_a (np.ndarray): State a described by its amplitudes.
        state_b (np.ndarray): State b described by its amplitudes.

    Returns:
        float: Estimation of the fidelity between the states.

        .. note::
           As it can be deduced from the expression, the probability of
           measuring :math:`\ket{0}` has a theoretical lower bound of
           :math:`0.5`, guaranteeing the lower bound for the fidelity of
           :math:`0`. In reality, the probability of measurement is estimated
           by sampling. This causes an imprecision in the value obtained,
           sometimes dipping the fidelity value below :math:`0`. In order to
           avoid problems with other processes based on fidelity estimation,
           the value returned has been forced to be positive, changing it to
           :math:`0` if it is negative.
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

    return max(2.0 * result.get_counts(comp)['0'] / shots - 1.0, 0)


def distance_estimation(
        a: np.ndarray,
        a_norm: float,
        b: np.ndarray,
        b_norm: float,
) -> float:
    r"""Euclidean distance estimation through fidelity estimation.
    :cite:`lloyd2013quantum`

    The Euclidean distance between two vectors :math:`\boldsymbol{a}` and
    :math:`\boldsymbol{b}` can be estimated through the fidelity
    between two quantum states by following these steps:

    1. Initialize two quantum
    states:

    .. math::
       \ket{\psi} &= \frac{1}{\sqrt{2}}(\ket{0,\boldsymbol{a}}+
       \ket{1,\boldsymbol{b}}) \\
       \ket{\phi} &= \frac{1}{\sqrt{Z}}(|\boldsymbol{a}|\ket{0}-
       |\boldsymbol{b}|\ket{1})

    Where :math:`Z=|\boldsymbol{a}|+|\boldsymbol{b}|` and :math:`\ket{a}` is
    constructed using amplitude encoding:

    .. math::
       \ket{\boldsymbol{a}} = \sum_{i=1}^{N}|\boldsymbol{a}|^{-1}a_i\ket{i}

    This will define an inner product:

    .. math::
       \braket{\phi|\psi} &= \frac{1}{\sqrt{2Z}}(|\boldsymbol{a}|
       \braket{0|0,\boldsymbol{a}} + |\boldsymbol{a}|
       \braket{0|1,\boldsymbol{b}} - |\boldsymbol{b}|
       \braket{1|0,\boldsymbol{a}} - |\boldsymbol{b}|
       \braket{1|1,\boldsymbol{b}}) \\
                          &= \frac{1}{\sqrt{2Z}}(|\boldsymbol{a}|
                          \ket{\boldsymbol{a}} - |\boldsymbol{b}|
                          \ket{\boldsymbol{b}})

    Because the quantum states do not have the same amount of qubits, the
    inner product can be interpreted as a projection resulting in another
    quantum state, instead of a numerical value.

    2. Evaluate the fidelity
    of the two states:

    .. math::
       |\braket{\phi|\psi}|^2 &= \braket{\phi|\psi}\cdot\braket{\phi|\psi} \\
                              &= \frac{1}{2Z}(|\boldsymbol{a}|^2 +
       |\boldsymbol{b}|^2 - |\boldsymbol{a}||\boldsymbol{b}|
       \braket{\boldsymbol{a}|\boldsymbol{b}} - |\boldsymbol{a}||\boldsymbol{b}
       |\braket{\boldsymbol{b}|\boldsymbol{a}})

    Taking into account that:

    .. math::
       \braket{\boldsymbol{b}|\boldsymbol{a}}=
       \braket{\boldsymbol{a}|\boldsymbol{b}}^* =
       \braket{\boldsymbol{a}|\boldsymbol{b}}

    With :math:`^*` being the complex conjugate. This is possible because the
    input vectors only have real values, therefore, the quantum state's
    amplitudes are also real.

    .. math::
       |\braket{\phi|\psi}|^2 &= \frac{1}{2Z}(|\boldsymbol{a}|^2 +
       |\boldsymbol{b}|^2 - 2|\boldsymbol{a}||\boldsymbol{b}|
       \braket{\boldsymbol{a}|\boldsymbol{b}}) \\
       &= \frac{1}{2Z}(|\boldsymbol{a}|^2 +
       |\boldsymbol{b}|^2 - 2\boldsymbol{a}^T\cdot\boldsymbol{b}) \\
       &= \frac{1}{2Z}|\boldsymbol{a}-\boldsymbol{b}|^2

    Args:
        a (np.ndarray): Input a.
        a_norm (float): L2-norm of input a.
        b (np.ndarray): Input b.
        b_norm (float): L2-norm of input b.

    Returns:
        float: Euclidean distance estimated.
    """
    if a.shape != b.shape:
        raise ValueError(f'Vector dimensions disparity between {a.shape} '
                         'and {b.shape}')

    # Padding with 0s in order to force the quantum states to be whole qubit
    # sized
    size = max(int(2 ** np.ceil(np.log2(a.shape[0]))), 2)
    a = np.pad(a, (0, size - a.shape[0]))
    b = np.pad(b, (0, size - b.shape[0]))

    # Definition of quantum state whose fidelity relates to the distance
    z = a_norm ** 2 + b_norm ** 2
    phi = np.array([a_norm, -b_norm]) / np.sqrt(z)
    psi = np.concatenate([a / a_norm, b / b_norm]) / np.sqrt(2.0)

    fidelity = fidelity_estimation(phi, psi)
    return np.sqrt(2.0 * z * fidelity)


def inner_product_estimation(
        state_a: np.ndarray,
        state_b: np.ndarray,
) -> float:
    r"""Quantum estimation of the inner product between two quantum
    states. :cite:`algebraquantum2019`

    The inner product between two vectors :math:`\boldsymbol{a},\boldsymbol{b}`
    :math:`\in \mathbb{R}^N` can be estimated with the following steps:

    1. Initialize the
    quantum state:

    .. math::
       \ket{\psi_0} = \frac{1}{\sqrt{2}}(\ket{0,\boldsymbol{a}} +
       \ket{1, \boldsymbol{b}})

    Supposing the input vectors are normalized and the quantum states defined
    as :math:`\ket{\boldsymbol{a}}=\sum_{i=1}^Na_i\ket{i}`

    2. Apply a Hadamard gate to the ancilla qubit:

    .. math::
       \ket{\psi_1} &= (H \bigotimes I^{\bigotimes n})\ket{\psi_0} \\
       &= \frac{1}{2}(\ket{0}(\ket{\boldsymbol{a}} + \ket{\boldsymbol{b}}) +
       \ket{1}(\ket{\boldsymbol{a}} - \ket{\boldsymbol{b}}))

    3. Estimate the probability of measuring :math:`\ket{0}`:

    .. math::
       P(\ket{0}) &= |\braket{0|\psi_1}|^2 \\
       &= |\frac{1}{2}(\braket{0|0}(\ket{\boldsymbol{a}} +
       \ket{\boldsymbol{b}}) + \braket{0|1}(\ket{\boldsymbol{a}} -
       \ket{\boldsymbol{b}}))|^2 \\
       &= |\frac{1}{2}(\ket{\boldsymbol{a}} + \ket{\boldsymbol{b}})|^2 \\
       &= \frac{1}{4}\sum_{i=1}^N(a_i+b_i)^2 \\
       &= \frac{1}{4}\braket{\boldsymbol{a}+\boldsymbol{b}|
       \boldsymbol{a}+\boldsymbol{b}} \\
       &= \frac{1}{4}(\braket{\boldsymbol{a}|\boldsymbol{a}} +
       \braket{\boldsymbol{a}|\boldsymbol{b}} +
       \braket{\boldsymbol{b}|\boldsymbol{a}} +
       \braket{\boldsymbol{b}|\boldsymbol{b}}) \\
       &= \frac{1}{4} (2 + 2\braket{\boldsymbol{a}|\boldsymbol{b}})

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
