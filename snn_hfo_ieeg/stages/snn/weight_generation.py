import random
import numpy as np
POSSIBLE_ABSOLUTE_WEIGHTS = [1000, 2000]


def _generate_weights_for_input_pair(hidden_neuron_count):
    weights = random.choices(POSSIBLE_ABSOLUTE_WEIGHTS,
                             k=int(hidden_neuron_count))
    half_point = hidden_neuron_count // 2
    first_input_to_first_half_of_hidden = weights[:half_point]
    first_input_to_second_half_of_hidden = weights[half_point:] * -1
    second_input_to_first_half_of_hidden = weights[:half_point] * -1
    second_input_to_second_half_of_hidden = weights[half_point:]
    return np.concatenate([
        first_input_to_first_half_of_hidden,
        first_input_to_second_half_of_hidden,
        second_input_to_first_half_of_hidden,
        second_input_to_second_half_of_hidden])


def generate_weights(input_neuron_count, hidden_neuron_count):
    input_pair_count = input_neuron_count // 2
    weights = [_generate_weights_for_input_pair(
        _) for _ in [hidden_neuron_count] * input_pair_count]
    return np.concatenate(weights)
