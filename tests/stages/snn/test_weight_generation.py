import pytest
import numpy as np
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights, POSSIBLE_ABSOLUTE_WEIGHTS
from utility import quarter

INPUT_TEST_PAIRS = [(2, 2),
                    (2, 4),
                    (4, 16),
                    (16, 2)]


@pytest.mark.parametrize(
    'input_neuron_count, hidden_neuron_count',
    INPUT_TEST_PAIRS
)
def test_weights_sum_to_zero(input_neuron_count, hidden_neuron_count):
    weights = generate_weights(input_neuron_count, hidden_neuron_count)
    assert sum(weights) == 0


@pytest.mark.parametrize(
    'input_neuron_count, hidden_neuron_count',
    INPUT_TEST_PAIRS
)
def test_weights_are_in_range(input_neuron_count, hidden_neuron_count):
    weights = generate_weights(input_neuron_count, hidden_neuron_count)
    min_weight = min(POSSIBLE_ABSOLUTE_WEIGHTS)
    max_weight = max(POSSIBLE_ABSOLUTE_WEIGHTS)
    assert np.all(min_weight <= weights <= max_weight)


@pytest.mark.parametrize(
    'input_neuron_count, hidden_neuron_count',
    INPUT_TEST_PAIRS
)
def test_weights_have_expected_length(input_neuron_count, hidden_neuron_count):
    weights = generate_weights(input_neuron_count, hidden_neuron_count)
    # Times 2 because one is inhibitory, one is excitatory
    expected_length = input_neuron_count // 2 * hidden_neuron_count * 2
    assert len(weights) == expected_length


def test_weights_have_correct_sequence():
    weights = generate_weights(2, 4)
    quarters = quarter(weights)
    assert np.all(quarters.first > 0)
    assert np.all(quarters.second < 0)
    assert np.all(quarters.third < 0)
    assert np.all(quarters.fourth > 0)


def test_weight_generation_fails_on_odd_number_of_inputs():
    with pytest.raises(ValueError):
        generate_weights(3, 2)


def test_weight_generation_fails_on_odd_number_of_hidden_neurons():
    with pytest.raises(ValueError):
        generate_weights(2, 3)
