import numpy as np
import pytest
from snn_hfo_ieeg.stages.snn.tau_generation import generate_taus, MIN_TAU, MAX_TAU, MIN_DELTA_TAU, MAX_DELTA_TAU

ARBITRARY_BIG_NUMBER = 1000000
ARBITRARY_ACCURACY = 0.01
EXPECTED_OUTLIER_FRACTION = 0.05


def _generate_test_taus():
    return generate_taus(ARBITRARY_BIG_NUMBER)


def _get_stochastic_inaccuracy_for_mean_in_range(taus, min, max):
    expected_mean = np.mean([min, max])
    actual_mean = np.mean(taus)
    return abs(expected_mean - actual_mean)


def test_tau_generation_has_right_excitatory_mean():
    excitatory_taus, _ = _generate_test_taus()
    stochastic_inaccuracy = _get_stochastic_inaccuracy_for_mean_in_range(
        excitatory_taus, MIN_TAU, MAX_TAU)
    assert stochastic_inaccuracy < ARBITRARY_ACCURACY


def test_tau_generation_has_right_inhibitory_mean():
    _, inhibitory_taus = _generate_test_taus()
    mean_delta = np.mean([MIN_DELTA_TAU, MAX_DELTA_TAU])
    stochastic_inaccuracy = _get_stochastic_inaccuracy_for_mean_in_range(
        inhibitory_taus, MIN_TAU - mean_delta, MAX_TAU - mean_delta)
    assert stochastic_inaccuracy < ARBITRARY_ACCURACY


def _get_stochastic_inaccuracy_for_taus_outside_range(taus, min, max):
    outliers = [tau for tau in taus if tau <
                min or tau > max]
    actual_outlier_fraction = len(outliers) / len(taus)
    return actual_outlier_fraction - EXPECTED_OUTLIER_FRACTION


def test_tau_generation_has_right_excitatory_bounds():
    excitatory_taus, _ = _generate_test_taus()
    stochastic_inaccuracy = _get_stochastic_inaccuracy_for_taus_outside_range(
        excitatory_taus, MIN_TAU, MAX_TAU)
    assert stochastic_inaccuracy < ARBITRARY_ACCURACY


def test_tau_generation_has_right_inhibitory_bounds():
    _, inhibitory_taus = _generate_test_taus()
    stochastic_inaccuracy = _get_stochastic_inaccuracy_for_taus_outside_range(
        inhibitory_taus, MIN_TAU-MAX_DELTA_TAU, MAX_TAU+MAX_DELTA_TAU)
    assert stochastic_inaccuracy < ARBITRARY_ACCURACY


def test_inhibitory_mean_is_maller_than_excitatory():
    excitatory_taus, inhibitory_taus = _generate_test_taus()
    excitatory_mean = np.mean(excitatory_taus)
    inhibitory_mean = np.mean(inhibitory_taus)
    assert inhibitory_mean < excitatory_mean


def test_tau_generation_fails_on_odd_number():
    with pytest.raises(ValueError):
        generate_taus(ARBITRARY_BIG_NUMBER + 1)


def test_taus_have_specified_size():
    excitatory_taus, inhibitory_taus = _generate_test_taus()
    assert len(excitatory_taus) == len(inhibitory_taus) == ARBITRARY_ACCURACY
