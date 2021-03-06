import pytest
from snn_hfo_detection.functions.dynapse_biases import *
from tests.utility import *


@pytest.mark.parametrize(
    'current,expected_tau',
    [(1, 5.35e-14),
     (1e-4, 5.35e-10),
     (1.5e-4, 3.57e-10),
     (-1, -5.35e-14)]
)
def test_get_tau(current, expected_tau):
    actual_tau = get_tau(current)
    assert expected_tau == pytest.approx(actual_tau)


def test_get_tau_raises_error_on_zero_current():
    with pytest.raises(ZeroDivisionError):
        get_tau(0)


@pytest.mark.parametrize(
    'tau,expected_current',
    [(1, 5.35e-14),
     (1e-4, 5.35e-10),
     (1.5e-4, 3.57e-10),
     (-1, -5.35e-14), ]
)
def test_get_current(tau, expected_current):
    actual_current = get_current(tau)
    assert expected_current == pytest.approx(actual_current)


def test_get_current_raises_error_on_zero_current():
    with pytest.raises(ZeroDivisionError):
        get_current(0)


@pytest.mark.parametrize(
    'tau,expected_current',
    [(np.array([1, 1e-4]), [5.35e-14, 5.35e-10]),
     (np.array([1.5e-4, -1]), [3.57e-10, -5.35e-14])
     ]
)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_get_current_with_vector(tau, expected_current):
    actual_current = get_current(tau)
    assert_are_lists_approximately_equal(actual_current, expected_current)
