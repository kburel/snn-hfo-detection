import numpy as np


def _is_odd(number):
    return number % 2 != 0 or number < 2


def _concatenate_excitatory_and_inhibitory_for_input_pair(hidden_neuron_count, generate_array_cb):
    excitatory_array, inhibitory_array = generate_array_cb(hidden_neuron_count)
    print(excitatory_array)
    print(inhibitory_array)
    half_point = len(excitatory_array) // 2
    first_input_to_first_half_of_hidden = excitatory_array[:half_point]
    first_input_to_second_half_of_hidden = inhibitory_array[:half_point]
    second_input_to_first_half_of_hidden = inhibitory_array[half_point:]
    second_input_to_second_half_of_hidden = excitatory_array[half_point:]
    return np.concatenate([
        first_input_to_first_half_of_hidden,
        first_input_to_second_half_of_hidden,
        second_input_to_first_half_of_hidden,
        second_input_to_second_half_of_hidden])


def concatenate_excitatory_and_inhibitory_with_generator_function(input_neuron_count, hidden_neuron_count, generate_array_cb):
    if _is_odd(input_neuron_count):
        raise ValueError(
            f'input_neuron_count must be a positive, even number but was odd. Actual value: {input_neuron_count}')
    if _is_odd(hidden_neuron_count):
        raise ValueError(
            f'hidden_neuron_count must be a positive, even number but was odd. Actual value: {hidden_neuron_count}')
    input_pair_count = input_neuron_count // 2
    weights = [_concatenate_excitatory_and_inhibitory_for_input_pair(_, generate_array_cb)
               for _
               in [hidden_neuron_count] * input_pair_count]
    return np.concatenate(weights)
