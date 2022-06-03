import pytest
import numpy as np
import random
from skqlearn.ml.clustering import KMedians, KMeans
from skqlearn.utils import JobHandler
from qiskit.providers.aer import AerSimulator


@pytest.mark.parametrize('n, iterations, state, calc_method',
                         [
                             [5, 15, 0, 'normal'],
                             [5, 15, 'test', 'classic'],
                             [5, -5, 2, 'quantum'],
                             [0, 15, 3, 'classic'],
                         ])
def test_kmeans_invalid_inputs(n, iterations, state, calc_method):
    with pytest.raises(ValueError):
        KMeans(n, iterations, state, calc_method)


@pytest.mark.parametrize('n, iterations, state, calc_method',
                         [
                             [5, 15, 0, 'normal'],
                             [5, 15, 'test', 'classic'],
                             [5, -5, 2, 'quantum'],
                             [0, 15, 3, 'classic'],
                         ])
def test_kmedians_invalid_inputs(n, iterations, state, calc_method):
    with pytest.raises(ValueError):
        KMedians(n, iterations, state, calc_method)


@pytest.mark.parametrize('n, iterations, state, calc_method',
                         [
                             [np.random.randint(1, 100),
                              np.random.randint(1, 100),
                              random.choice([np.random.randint(1, 100),
                                             np.random.RandomState()]),
                              random.choice(['classic', 'quantum'])]
                             for _ in range(10)
                         ])
def test_kmeans_valid_inputs(n, iterations, state, calc_method):
    KMeans(n, iterations, state, calc_method)


@pytest.mark.parametrize('n, iterations, state, calc_method',
                         [
                             [np.random.randint(1, 100),
                              np.random.randint(1, 100),
                              random.choice([np.random.randint(1, 100),
                                             np.random.RandomState()]),
                              random.choice(['classic', 'quantum'])]
                             for _ in range(10)
                         ])
def test_kmedians_valid_inputs(n, iterations, state, calc_method):
    KMedians(n, iterations, state, calc_method)


def test_kmeans_classic_execution():
    x = np.array([[0, 10], [0, 10.1], [10, 0], [10.1, 0]])
    result = KMeans(2, 10, None, 'classic').fit_predict(x)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert result[0] == result[1] and result[2] == result[3]


def test_kmeans_quantum_execution():
    JobHandler().configure(AerSimulator(), 50000)
    x = np.array([[0, 10], [0, 10.1], [10, 0], [10.1, 0]])
    result = KMeans(2, 10, None, 'quantum').fit_predict(x)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert result[0] == result[1] and result[2] == result[3]


def test_kmedians_classic():
    x = np.array([[0, 10], [0, 10.1], [10, 0], [10.1, 0]])
    result = KMedians(2, 10, None, 'classic').fit_predict(x)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert result[0] == result[1] and result[2] == result[3]


def test_kmedians_quantum():
    JobHandler().configure(AerSimulator(), 50000)
    x = np.array([[0, 10], [0, 10.1], [10, 0], [10.1, 0]])
    result = KMedians(2, 10, None, 'quantum').fit_predict(x)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert result[0] == result[1] and result[2] == result[3]
