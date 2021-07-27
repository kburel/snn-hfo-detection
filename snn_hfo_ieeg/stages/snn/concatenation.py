from typing import NamedTuple
import numpy as np


class NeuronCount(NamedTuple):
    input: int
    hidden: int


def _is_odd(number):
    return number % 2 != 0 or number < 2


def _concatenate_excitatory_and_inhibitory_for_input_pair(hidden_neuron_count, generate_array_cb):
    excitatory_array, inhibitory_array = generate_array_cb(hidden_neuron_count)
    half_point = len(excitatory_array) // 2
    first_input_to_first_half_of_hidden = excitatory_array[:half_point]
    first_input_to_second_half_of_hidden = inhibitory_array[half_point:]
    second_input_to_first_half_of_hidden = inhibitory_array[:half_point]
    second_input_to_second_half_of_hidden = excitatory_array[half_point:]
    return np.concatenate([
        first_input_to_first_half_of_hidden,
        first_input_to_second_half_of_hidden,
        second_input_to_first_half_of_hidden,
        second_input_to_second_half_of_hidden])


def concatenate_excitatory_and_inhibitory_with_generator_function(neuron_count, generate_array_cb):
    if _is_odd(neuron_count.input):
        raise ValueError(
            f'input_neuron_count must be a positive, even number, but got: {neuron_count.input}')
    if _is_odd(neuron_count.hidden):
        raise ValueError(
            f'hidden_neuron_count must be a positive, even number, but got: {neuron_count.hidden}')
    input_pair_count = neuron_count.input // 2
    weights = [_concatenate_excitatory_and_inhibitory_for_input_pair(_, generate_array_cb)
               for _
               in [neuron_count.hidden] * input_pair_count]
    return np.concatenate(weights)
