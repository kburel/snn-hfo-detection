import numpy as np
import pytest
from tests.stages.snn.utility import quarter
from snn_hfo_ieeg.stages.snn.tau_generation import generate_taus, generate_concatenated_taus, OUTLIER_FRACTION, MIN_TAU, MAX_TAU, MIN_DELTA_TAU, MAX_DELTA_TAU
from snn_hfo_ieeg.stages.snn.concatenation import NeuronCount

ARBITRARY_BIG_NUMBER = 1000000
ARBITRARY_ACCURACY = 0.01


def _generate_test_taus():
    return generate_taus(ARBITRARY_BIG_NUMBER)


def _get_stochastic_inaccuracy_for_mean_in_range(taus, min, max):
    expected_mean = np.mean([min, max])
    actual_mean = np.mean(taus)
    return abs(expected_mean - actual_mean)


def _assert_excitatory_mean(taus):
    stochastic_inaccuracy = _get_stochastic_inaccuracy_for_mean_in_range(
        taus, MIN_TAU, MAX_TAU)
    assert stochastic_inaccuracy < ARBITRARY_ACCURACY


def test_tau_generation_has_right_excitatory_mean():
    excitatory_taus, _ = _generate_test_taus()
    _assert_excitatory_mean(excitatory_taus)


def _assert_inhibitory_mean(taus):
    mean_delta = np.mean([MIN_DELTA_TAU, MAX_DELTA_TAU])
    stochastic_inaccuracy = _get_stochastic_inaccuracy_for_mean_in_range(
        taus, MIN_TAU - mean_delta, MAX_TAU - mean_delta)
    assert stochastic_inaccuracy < ARBITRARY_ACCURACY


def test_tau_generation_has_right_inhibitory_mean():
    _, inhibitory_taus = _generate_test_taus()
    _assert_inhibitory_mean(inhibitory_taus)


def _get_stochastic_inaccuracy_for_taus_outside_range(taus, min, max):
    outliers = [tau for tau in taus if tau <
                min or tau > max]
    actual_outlier_fraction = len(outliers) / len(taus)
    return actual_outlier_fraction - OUTLIER_FRACTION


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


def test_taus_have_specified_size():
    excitatory_taus, inhibitory_taus = _generate_test_taus()
    assert len(excitatory_taus) == len(inhibitory_taus) == ARBITRARY_BIG_NUMBER


def test_inhibitory_taus_are_always_smaller_than_excitatory_ones():
    excitatory_taus, inhibitory_taus = _generate_test_taus()
    for excitatory_tau, inhibitory_tau in zip(excitatory_taus, inhibitory_taus):
        assert excitatory_tau > inhibitory_tau


def test_concatenated_tau_generation_fails_on_odd_number_of_inputs():
    with pytest.raises(ValueError):
        generate_concatenated_taus(
            NeuronCount(input=ARBITRARY_BIG_NUMBER + 1, hidden=ARBITRARY_BIG_NUMBER))


def test_concatenated_tau_generation_fails_on_odd_number_of_hidden_neurons():
    with pytest.raises(ValueError):
        generate_concatenated_taus(
            NeuronCount(input=ARBITRARY_BIG_NUMBER, hidden=ARBITRARY_BIG_NUMBER+1))


@pytest.mark.parametrize(
    'neuron_count',
    [NeuronCount(input=2, hidden=2),
     NeuronCount(input=2, hidden=4),
     NeuronCount(input=4, hidden=16),
     NeuronCount(input=16, hidden=2)]
)
def test_concatenated_tau_generation_has_correct_length(neuron_count):
    taus = generate_concatenated_taus(neuron_count)
    # Times 2 because one is inhibitory, one is excitatory
    expected_length = neuron_count.input // 2 * neuron_count.hidden * 2
    assert len(taus) == expected_length


def test_concatenated_tau_generation_has_right_sequence_for_input_pair():
    taus = generate_concatenated_taus(
        NeuronCount(input=2, hidden=ARBITRARY_BIG_NUMBER))
    quarters = quarter(taus)

    _assert_excitatory_mean(quarters.first)
    _assert_inhibitory_mean(quarters.second)
    _assert_inhibitory_mean(quarters.third)
    _assert_excitatory_mean(quarters.fourth)
