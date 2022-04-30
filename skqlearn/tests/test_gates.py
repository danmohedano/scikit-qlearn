import pytest
from skqlearn.gates import multiqubit_cswap
import qiskit


def test_multi_cswap():
    """Test the correct creation of multi-qubit CSWAP"""
    assert isinstance(multiqubit_cswap(2, 2), qiskit.circuit.Instruction)


def test_multi_cswap_incorrect():
    """Test ValueError raised when provided invalid qubit sizes"""
    with pytest.raises(ValueError):
        multiqubit_cswap(-2, 2)
        multiqubit_cswap(2, -2)
