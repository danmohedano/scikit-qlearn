import qiskit
from typing import Union


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

    All quantum processes implemented in the library execute the quantum
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
            backend: qiskit.providers.Backend,
            shots: int,
            run_options: dict = {},
    ):
        """Configuration of the job handler.

        Args:
            backend (qiskit.providers.Backend): Desired backend to use when
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
            circuits: Union[qiskit.circuit.QuantumCircuit, list]
    ) -> qiskit.result.Result:
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
