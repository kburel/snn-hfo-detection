import pytest
import numpy as np
from tests.stages.snn.utility import quarter
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights, POSSIBLE_ABSOLUTE_WEIGHTS
from snn_hfo_ieeg.stages.snn.concatenation import NeuronCount

TEST_NEURON_COUNT = [NeuronCount(input=2, hidden=2),
                     NeuronCount(input=2, hidden=4),
                     NeuronCount(input=4, hidden=16),
                     NeuronCount(input=16, hidden=2)]


@pytest.mark.parametrize(
    'input_neuron_count, hidden_neuron_count',
    TEST_NEURON_COUNT
)
def test_weights_sum_to_zero(input_neuron_count, hidden_neuron_count):
    weights = generate_weights(NeuronCount(
        input=input_neuron_count, hidden=hidden_neuron_count))
    assert sum(weights) == 0


@pytest.mark.parametrize(
    'input_neuron_count, hidden_neuron_count',
    TEST_NEURON_COUNT
)
def test_weights_are_in_range(input_neuron_count, hidden_neuron_count):
    weights = generate_weights(NeuronCount(
        input=input_neuron_count, hidden=hidden_neuron_count))
    min_weight = min(POSSIBLE_ABSOLUTE_WEIGHTS)
    max_weight = max(POSSIBLE_ABSOLUTE_WEIGHTS)
    for weight in weights:
        assert min_weight <= weight <= max_weight or -max_weight <= weight <= -min_weight


@pytest.mark.parametrize(
    'neuron_count',
    TEST_NEURON_COUNT
)
def test_weights_have_expected_length(neuron_count):
    weights = generate_weights(neuron_count)
    # Times 2 because one is inhibitory, one is excitatory
    expected_length = neuron_count.input // 2 * neuron_count.hidden * 2
    assert len(weights) == expected_length


def test_weights_have_correct_sequence():
    weights = generate_weights(NeuronCount(input=2, hidden=4))
    quarters = quarter(weights)
    assert np.all(quarters.first > 0)
    assert np.all(quarters.second < 0)
    assert np.all(quarters.third < 0)
    assert np.all(quarters.fourth > 0)


def test_weights_have_correct_sequence_with_multiple_input_pairs():
    weights = generate_weights(NeuronCount(input=4, hidden=32))
    half_point = len(weights) // 2
    for weight_half in [weights[:half_point], weights[half_point:]]:
        quarters = quarter(weight_half)
        assert np.all(quarters.first > 0)
        assert np.all(quarters.second < 0)
        assert np.all(quarters.third < 0)
        assert np.all(quarters.fourth > 0)


def test_weight_generation_fails_on_odd_number_of_inputs():
    with pytest.raises(ValueError):
        generate_weights(NeuronCount(input=3, hidden=2))


def test_weight_generation_fails_on_odd_number_of_hidden_neurons():
    with pytest.raises(ValueError):
        generate_weights(NeuronCount(input=2, hidden=3))
