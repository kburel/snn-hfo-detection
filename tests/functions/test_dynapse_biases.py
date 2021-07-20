import pytest
from SNN_HFO_iEEG.Functions.Dynapse_biases_functions import *

@pytest.mark.parametrize(
    "current,expected_tau",
    [(1, 5.35e-14),
    (1e-4, 5.35e-10),
    (1.5e-4, 3.57e-10),
    (-1, -5.35e-14)]
)
def test_get_tau(current, expected_tau):
    actual_tau = getTau(current)
    assert expected_tau == pytest.approx(actual_tau)

def test_get_tau_raises_error_on_zero_current():
    with pytest.raises(ZeroDivisionError):
         getTau(0)

@pytest.mark.parametrize(
    "tau,expected_current",
    [(1, 5.35e-14),
    (1e-4, 5.35e-10),
    (1.5e-4, 3.57e-10),
    (-1, -5.35e-14),
    (0, 2.390625e-05)] # Only difference to get_tau
)
def test_get_tau_current(tau, expected_current):
    actual_current = getTauCurrent(tau)
    assert expected_current == pytest.approx(actual_current)

def test_get_tau_current_raises_error_when_vector_is_specified():
    with pytest.raises(TypeError):
         getTauCurrent(1, vector=True)

@pytest.mark.parametrize(
    "mean_tau,std_tau,expected_means",
    [(0, 0, (2.390625e-5, 0, 2.390625e-5, 2.390625e-5)),
    (1, 1, (5.357142857142858e-11, 1.1953111607142857e-05, 2.678571428571429e-11, 2.390625e-05)),
    ]
)
# Seems to be unused...
def test_get_mean_std_currents(mean_tau,std_tau,expected_means):
    actual_means = get_mean_std_currents(mean_tau, std_tau)
    assert expected_means == pytest.approx(actual_means)