import pytest
import numpy as np
from skqlearn.jobhandler import JobHandler
from skqlearn.utils import inner_product_estimation
from qiskit.providers.aer import AerSimulator


def test_job_handler_not_configured():
    JobHandler().configure(None, 1)
    with pytest.raises(ValueError):
        inner_product_estimation(np.array([1, 0]), np.array([0, 1]))


def test_job_handler_incorrect_shots():
    with pytest.raises(ValueError):
        JobHandler().configure(AerSimulator(), -1, {})


@pytest.mark.parametrize('backend',
                         [
                             AerSimulator(),
                         ])
def test_job_handler_backends(backend):
    JobHandler().configure(backend, 10)
    inner_product_estimation(np.array([1, 0]), np.array([0, 1]))
