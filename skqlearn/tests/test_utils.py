import pytest
import numpy as np
from skqlearn.utils import fidelity_estimation, distance_estimation, \
    inner_product_estimation, JobHandler
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


def test_distance_estimation_incorrect():
    with pytest.raises(ValueError):
        distance_estimation(np.array([0, 1]), 0.0, np.array([0, 2, 1]), 0.0)


@pytest.mark.parametrize('state_a, state_b',
                         [[np.random.rand(2),
                           np.random.rand(2)] for _ in range(50)])
def test_distance_estimation_correct(state_a, state_b):
    JobHandler().configure(AerSimulator(), 10000)
    result = distance_estimation(state_a, np.linalg.norm(state_a),
                                 state_b, np.linalg.norm(state_b))
    assert result >= 0


@pytest.mark.parametrize('state_a, state_b',
                         [[np.random.rand(2),
                           np.random.rand(2)] for _ in range(50)])
def test_fidelity_estimation(state_a, state_b):
    state_a /= np.linalg.norm(state_a)
    state_b /= np.linalg.norm(state_b)
    JobHandler().configure(AerSimulator(), 10000)
    result = fidelity_estimation(state_a, state_b)
    assert 0 <= result <= 1


@pytest.mark.parametrize('state_a, state_b',
                         [[np.random.rand(2),
                           np.random.rand(2)] for _ in range(50)])
def test_inner_product_estimation(state_a, state_b):
    state_a /= np.linalg.norm(state_a)
    state_b /= np.linalg.norm(state_b)
    JobHandler().configure(AerSimulator(), 10000)
    result = inner_product_estimation(state_a, state_b)
    assert -1 <= result <= 1
