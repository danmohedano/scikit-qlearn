import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend
from qiskit.result import Result
import qiskit
from typing import Union
from skqlearn.gates import multiqubit_cswap


class Singleton(type):
    """Singleton class implementation.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


class JobHandler(metaclass=Singleton):
    """Singleton class to contain the configuration for quantum executions.

    All quantum subroutines implemented in the package execute the quantum
    circuits through this handler. Therefore, before executing any of them,
    a backend should be correctly configured.

    Attributes:
        backend (qiskit.proviers.Backend): Backend where the quantum circuits
            will be run.
        shots (int): Amount of shots (repetitions) in the execution of the
            circuits.
        run_options (dict): Dictionary to provide the backend with extra
            options when executing the `run` method.
        compiled_circuits (Union[QuantumCircuit, List[QuantumCircuit]]):
            Compiled circuits in the last `run_job` batch.

    Examples:
        The JobHandler can be configured with local simulators.

        >>> from qiskit.providers.aer import AerSimulator
        >>> JobHandler().configure(AerSimulator(), 10000)

        And with remote systems and simulators accessed through your IBMQ
        account.

        >>> from qiskit import IBMQ
        >>> provider = IBMQ.enable_account('MY_API_TOKEN')
        >>> backend = provider.get_backend('ibmq_qasm_simulator')
        >>> JobHandler().configure(backend, 10000)
        >>> backend = provider.get_backend('ibmq_montreal')
        >>> JobHandler().configure(backend, 10000)
    """
    def __init__(self):
        self.backend = None
        self.run_options = {}
        self.shots = None
        self.compiled_circuits = None

    def configure(
            self,
            backend: Backend,
            shots: int,
            run_options: dict = {},
    ):
        """Configuration of the job handler.

        Args:
            backend (Backend): Desired backend to use when
                running circuits (either simulator or real backends).
            shots (int): Number of executions.
            run_options (dict): Keyword dictionary used in the run method as
                run time backend options.

        Raises:
            ValueError: If a non-positive amount of shots is provided.
        """
        if shots < 1:
            raise ValueError('Invalid value for shots provided. Expected '
                             f'positive integer, got {shots} instead.')
        self.backend = backend
        self.shots = shots
        self.run_options = run_options

    def run_job(
            self,
            circuits: Union[QuantumCircuit, list]
    ) -> Result:
        """Runs the provided circuit with the configured backend.

        Args:
            circuits (QuantumCircuit or list): Quantum Circuit(s) to run.

        Returns:
            Result:
                Result object.

        Raises:
            ValueError: If no backend has been previously configured.
        """
        if not self.backend:
            raise ValueError('Backend not configured in the JobHandler. Must '
                             'configured before trying to run quantum '
                             'operations.')

        self.compiled_circuits = qiskit.transpile(circuits, self.backend)
        job = self.backend.run(self.compiled_circuits,
                               shots=self.shots,
                               **self.run_options)

        return job.result()


def fidelity_estimation(
        state_a: np.ndarray,
        state_b: np.ndarray,
) -> float:
    r"""Fidelity estimation between two quantum states by
    SWAP-Test. :cite:`inaquantumworld`

    The fidelity of two quantum states can be defined as:

    .. math::
       Fid(\left|a\right>, \left|b\right>)=|\left<a|b\right>|^2

    This is a similarity measure between two quantum states, its value ranging
    from 0 if the states are orthogonal to 1 if they are identical.

    It can be measured by performing a SWAP-Test, with the following
    operations:

    1. Initialize the state with an ancilla qubit and the two n-qubit
    quantum states:

    .. math:: \left|\psi_0\right>=\left|0,a,b\right>

    2. Apply a Hadamard gate to the control
    qubit:

    .. math::
       \left|\psi_1\right> = (H \bigotimes I^{\bigotimes n} \bigotimes
       I^{\bigotimes n})\left|\psi_0\right> = \frac{1}{\sqrt{2}}(\left|0,a,b
       \right> + \left|1,a,b\right>)

    3. Apply a CSWAP gate to the three states, using the ancilla qubit as
    control:

    .. math::
       \left|\psi_2\right>= \frac{1}{\sqrt{2}}(\left|0,a,b\right> +
       \left|1,b,a\right>)

    4. Apply a Hadamard gate to the control
    qubit:

    .. math::
       \left|\psi_3\right> = \frac{1}{2}\left|0\right>(\left|a,b\right>+
       \left|b,a\right>) + \frac{1}{2} \left|1\right>(\left|a,b\right>
       -\left|b,a\right>)

    5. Measure the probability of the control qubit being
    in state :math:`\left|0\right>`:

    .. math::
       P(\left|0\right>) &= |\left<0|\psi_3\right>|^2 \\
                  &= \frac{1}{4}|(\left|a,b\right>+\left|b,a\right>)|^2 \\
                  &= \frac{1}{2} + \frac{1}{2}|\left<a|b\right>|^2

    The algorithm also works for quantum states with different sizes, though
    the notion of similiarity then becomes harder to define.

    Args:
        state_a (numpy.ndarray): State a described by its amplitudes.
        state_b (numpy.ndarray): State b described by its amplitudes.

    Returns:
        float:
            Estimation of the fidelity between the states.

        .. note::
           As it can be deduced from the expression, the probability of
           measuring :math:`\left|0\right>` has a theoretical lower bound of
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

    try:
        return max(2.0 * result.get_counts(comp)['0'] / shots - 1.0, 0)
    except KeyError:
        return -1.0


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
       \left|\psi\right> &= \frac{1}{\sqrt{2}}(\left|0,\boldsymbol{a}\right>+
       \left|1,\boldsymbol{b}\right>) \\
       \left|\phi\right> &= \frac{1}{\sqrt{Z}}(|\boldsymbol{a}|\left|0\right>-
       |\boldsymbol{b}|\left|1\right>)

    Where :math:`Z=|\boldsymbol{a}|+|\boldsymbol{b}|` and
    :math:`\left|a\right>` is constructed using amplitude encoding:

    .. math::
       \left|\boldsymbol{a}\right> = \sum_{i=1}^{N}|\boldsymbol{a}|^{-1}
       a_i\left|i\right>

    This will define an inner product:

    .. math::
       \left<\phi|\psi\right> &= \frac{1}{\sqrt{2Z}}(|\boldsymbol{a}|
       \left<0|0,\boldsymbol{a}\right> + |\boldsymbol{a}|
       \left<0|1,\boldsymbol{b}\right> - |\boldsymbol{b}|
       \left<1|0,\boldsymbol{a}\right> - |\boldsymbol{b}|
       \left<1|1,\boldsymbol{b}\right>) \\
                          &= \frac{1}{\sqrt{2Z}}(|\boldsymbol{a}|
                          \left|\boldsymbol{a}\right> - |\boldsymbol{b}|
                          \left|\boldsymbol{b}\right>)

    Because the quantum states do not have the same amount of qubits, the
    inner product can be interpreted as a projection resulting in another
    quantum state, instead of a numerical value.

    2. Evaluate the fidelity
    of the two states:

    .. math::
       |\left<\phi|\psi\right>|^2 &= \left<\phi|\psi\right>\cdot\left<
       \phi|\psi\right> \\
                              &= \frac{1}{2Z}(|\boldsymbol{a}|^2 +
       |\boldsymbol{b}|^2 - |\boldsymbol{a}||\boldsymbol{b}|
       \left<\boldsymbol{a}|\boldsymbol{b}\right> - |\boldsymbol{a}||
       \boldsymbol{b}
       |\left<\boldsymbol{b}|\boldsymbol{a}\right>)

    Taking into account that:

    .. math::
       \left<\boldsymbol{b}|\boldsymbol{a}\right>=
       \left<\boldsymbol{a}|\boldsymbol{b}\right>^* =
       \left<\boldsymbol{a}|\boldsymbol{b}\right>

    With :math:`^*` being the complex conjugate. This is possible because the
    input vectors only have real values, therefore, the quantum state's
    amplitudes are also real.

    .. math::
       |\left<\phi|\psi\right>|^2 &= \frac{1}{2Z}(|\boldsymbol{a}|^2 +
       |\boldsymbol{b}|^2 - 2|\boldsymbol{a}||\boldsymbol{b}|
       \left<\boldsymbol{a}|\boldsymbol{b}\right>) \\
       &= \frac{1}{2Z}(|\boldsymbol{a}|^2 +
       |\boldsymbol{b}|^2 - 2\boldsymbol{a}^T\cdot\boldsymbol{b}) \\
       &= \frac{1}{2Z}|\boldsymbol{a}-\boldsymbol{b}|^2

    Args:
        a (numpy.ndarray): Input a.
        a_norm (float): L2-norm of input a.
        b (numpy.ndarray): Input b.
        b_norm (float): L2-norm of input b.

    Returns:
        float:
            Euclidean distance estimated.

    Raises:
        ValueError: If there is a dimension disparity between the vectors.
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
       \left|\psi_0\right> = \frac{1}{\sqrt{2}}(\left|0,\boldsymbol{a}\right> +
       \left|1, \boldsymbol{b}\right>)

    Supposing the input vectors are normalized and the quantum states defined
    as :math:`\left|\boldsymbol{a}\right>=\sum_{i=1}^Na_i\left|i\right>`

    2. Apply a Hadamard gate to the ancilla qubit:

    .. math::
       \left|\psi_1\right> &= (H \bigotimes I^{\bigotimes n})
       \left|\psi_0\right> \\
       &= \frac{1}{2}(\left|0\right>(\left|\boldsymbol{a}\right> +
       \left|\boldsymbol{b}\right>) + \left|1\right>(\left|
       \boldsymbol{a}\right> - \left|\boldsymbol{b}\right>))

    3. Estimate the probability of measuring :math:`\left|0\right>`:

    .. math::
       P(\left|0\right>) &= |\left<0|\psi_1\right>|^2 \\
       &= |\frac{1}{2}(\left<0|0\right>(\left|\boldsymbol{a}\right> +
       \left|\boldsymbol{b}\right>) + \left<0|1\right>
       (\left|\boldsymbol{a}\right> -
       \left|\boldsymbol{b}\right>))|^2 \\
       &= |\frac{1}{2}(\left|\boldsymbol{a}\right> +
       \left|\boldsymbol{b}\right>)|^2 \\
       &= \frac{1}{4}\sum_{i=1}^N(a_i+b_i)^2 \\
       &= \frac{1}{4}\left<\boldsymbol{a}+\boldsymbol{b}|
       \boldsymbol{a}+\boldsymbol{b}\right> \\
       &= \frac{1}{4}(\left<\boldsymbol{a}|\boldsymbol{a}\right> +
       \left<\boldsymbol{a}|\boldsymbol{b}\right> +
       \left<\boldsymbol{b}|\boldsymbol{a}\right> +
       \left<\boldsymbol{b}|\boldsymbol{b}\right>) \\
       &= \frac{1}{4} (2 + 2\left<\boldsymbol{a}|\boldsymbol{b}\right>)

    Args:
        state_a (numpy.ndarray): State a described by its amplitudes.
        state_b (numpy.ndarray): State b described by its amplitudes.

    Returns:
        float:
            Estimation of the inner product between both quantum states.
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

    circuit.h(quantum_register[-1])

    # Addition of measurement to classical bit register
    circuit.measure(quantum_register[-1], classical_register)

    # Execution of the circuit
    job_handler = JobHandler()
    result = job_handler.run_job(circuit)

    # Estimation of the probability
    shots = job_handler.shots
    comp = job_handler.compiled_circuits

    try:
        return 2.0 * result.get_counts(comp)['0'] / shots - 1.0
    except KeyError:
        return -1.0
