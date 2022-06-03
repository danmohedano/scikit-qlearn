import pytest
import numpy as np
from skqlearn.encoding import AmplitudeEncoding, AngleEncoding, \
    BasisEncoding, ExpandedAmplitudeEncoding, QSampleEncoding
from skqlearn.ml.kernels import SqueezingKernel


def regular_test_correct(x, y, expected_out, kernel_fn):
    assert np.isclose(kernel_fn(x, y), expected_out, atol=1e-10).all()


def regular_test_incorrect(x, y, kernel_fn):
    with pytest.raises(ValueError):
        kernel_fn(x, y)


class TestBasisEncodingKernel:
    @pytest.mark.parametrize('x, y',
                             [[np.array([[np.random.randint(10)]
                                         for _
                                         in range(np.random.randint(1, 5))]),
                               np.array([[np.random.randint(10)]
                                         for _
                                         in range(np.random.randint(1, 5))]),
                               ]
                              for _ in range(10)
                              ])
    def test(self, x, y):
        solution = x.flatten()[:, None] == y.flatten()
        regular_test_correct(x, y, solution, BasisEncoding().classic_kernel)


class TestAmplitudeEncodingKernel:
    def test(self):
        for _ in range(10):
            n_features, n_x, n_y = np.random.randint(1, 10, size=3)
            degree = np.random.randint(1, 5)

            x = np.array([np.random.rand(n_features) for _ in range(n_x)])
            y = np.array([np.random.rand(n_features) for _ in range(n_y)])

            solution = np.dot(x, y.T) ** degree
            encoding = AmplitudeEncoding(degree=degree)

            regular_test_correct(x, y, solution, encoding.classic_kernel)


class TestExpandedAmplitudeEncodingKernel:
    def test(self):
        for _ in range(10):
            n_features, n_x, n_y, c = np.random.randint(1, 10, size=4)
            degree = np.random.randint(1, 5)

            x = np.array([np.random.rand(n_features) for _ in range(n_x)])
            y = np.array([np.random.rand(n_features) for _ in range(n_y)])

            solution = (np.dot(x, y.T) + c ** 2) ** degree
            encoding = ExpandedAmplitudeEncoding(degree=degree, c=c)

            regular_test_correct(x, y, solution, encoding.classic_kernel)


class TestAngleEncodingKernel:
    def test(self):
        for _ in range(10):
            n_features, n_x, n_y = np.random.randint(1, 10, size=3)

            x = np.array([np.random.rand(n_features) for _ in range(n_x)])
            y = np.array([np.random.rand(n_features) for _ in range(n_y)])

            solution = np.zeros((n_x, n_y))
            for i in range(n_x):
                for j in range(n_y):
                    inner = 1
                    for k in range(n_features):
                        inner *= np.cos(x[i, k] - y[j, k])
                    solution[i, j] = inner

            encoding = AngleEncoding()

            regular_test_correct(x, y, solution, encoding.classic_kernel)
