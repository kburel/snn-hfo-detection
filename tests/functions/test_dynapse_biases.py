import pytest
from SNN_HFO_iEEG.Functions.Dynapse_biases_functions import *

@pytest.mark.parametrize(
    "current,expected_tau",
    [(1, 5.35e-14),
    (1e-4, 5.35e-10),
    (1.5e-4, 3.57e-10),
    pytest.param(0, None, marks=pytest.mark.xfail),
    pytest.param(-1, None, marks=pytest.mark.xfail)],
)
def test_tau_is_collected(current, expected_tau):
    assert expected_tau == pytest.approx(getTau(current))
