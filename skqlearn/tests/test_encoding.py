import pytest
import numpy as np
from skqlearn.encoding import AmplitudeEncoding, AngleEncoding, \
    BasisEncoding, ExpandedAmplitudeEncoding, QSampleEncoding


def regular_test_correct(value_in, expected_out, encoding_method):
    assert np.isclose(encoding_method.encoding(value_in),
                      expected_out, atol=1e-10).all()


def regular_test_incorrect(value_in, encoding_method):
    with pytest.raises(ValueError):
        encoding_method.encoding(value_in)


class TestAmplitudeEncoding:
    @pytest.fixture
    def encoding_method(self):
        return AmplitudeEncoding()

    @pytest.mark.parametrize('value_in, degree, expected_out',
                             [
                                 [np.array([0.5, 0.5, 0.5, 0.5]),
                                  1,
                                  np.array([0.5, 0.5, 0.5, 0.5])],
                                 [np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                                  2,
                                  np.array([0.5, 0.5, 0.5, 0.5])],
                             ])
    def test_correct(self, value_in, degree, expected_out, encoding_method):
        encoding_method.degree = degree
        regular_test_correct(value_in, expected_out, encoding_method)

    @pytest.mark.parametrize('value_in',
                             [
                                 -1,
                                 [1, 2, 3],
                                 np.array([0.5, 0.5])
                             ])
    def test_incorrect(self, value_in, encoding_method):
        regular_test_incorrect(value_in, encoding_method)


class TestAngleEncoding:
    @pytest.fixture
    def encoding_method(self):
        return AngleEncoding()

    @pytest.mark.parametrize('value_in, expected_out',
                             [
                                 [np.array([0]), np.array([1, 0])],
                                 [np.array([np.pi / 2]), np.array([0, 1])]
                             ])
    def test_correct(self, value_in, expected_out, encoding_method):
        regular_test_correct(value_in, expected_out, encoding_method)

    @pytest.mark.parametrize('value_in',
                             [
                                 -1,
                                 4.5,
                                 [1, 2, 3],
                             ])
    def test_incorrect(self, value_in, encoding_method):
        regular_test_incorrect(value_in, encoding_method)


class TestBasisEncoding:
    @pytest.fixture
    def encoding_method(self):
        return BasisEncoding()

    @pytest.mark.parametrize('value_in, expected_out',
                             [
                                 [1, np.array([0, 1])],
                                 [np.array([0, 1]),
                                  np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])]
                             ])
    def test_correct(self, value_in, expected_out, encoding_method):
        regular_test_correct(value_in, expected_out, encoding_method)

    @pytest.mark.parametrize('value_in',
                             [
                                 -1,
                                 4.5,
                                 [1, 2, 3],
                             ])
    def test_incorrect(self, value_in, encoding_method):
        regular_test_incorrect(value_in, encoding_method)


class TestExpandedAmplitudeEncoding:
    @pytest.fixture
    def encoding_method(self):
        return ExpandedAmplitudeEncoding()

    @pytest.mark.parametrize('value_in, degree, expected_out',
                             [
                                 [np.array([1.0, 1.0, 1.0]),
                                  1,
                                  np.array([0.5, 0.5, 0.5, 0.5])],
                                 [np.array([1.0]),
                                  2,
                                  np.array([0.5, 0.5, 0.5, 0.5])],
                             ])
    def test_correct(self, value_in, degree, expected_out, encoding_method):
        encoding_method.degree = degree
        regular_test_correct(value_in, expected_out, encoding_method)

    @pytest.mark.parametrize('value_in',
                             [
                                 -1,
                                 4.5,
                                 [1, 2, 3],
                             ])
    def test_incorrect(self, value_in, encoding_method):
        regular_test_incorrect(value_in, encoding_method)


class TestQSampleEncoding:
    @pytest.fixture
    def encoding_method(self):
        return QSampleEncoding()

    @pytest.mark.parametrize('value_in, expected_out',
                             [
                                 [np.array([0.5, 0.5]),
                                  np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])]
                             ])
    def test_correct(self, value_in, expected_out, encoding_method):
        regular_test_correct(value_in, expected_out, encoding_method)

    @pytest.mark.parametrize('value_in',
                             [
                                 -1,
                                 4.5,
                                 [1, 2, 3],
                             ])
    def test_incorrect(self, value_in, encoding_method):
        regular_test_incorrect(value_in, encoding_method)
