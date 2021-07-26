import numpy as np
from functools import reduce

MIN_TAU = 3
MAX_TAU = 6
MIN_DELTA_TAU = 0.3
MAX_DELTA_TAU = 1

POSSIBLE_WEIGHTS = [1000, 2000]


def _get_mean_and_standard_deviation_for_range(min, max):
    mean = np.mean([min, max])
    # 95% of data is within 2 standard deviations
    standard_deviation = (max - mean) / 2
    return mean, standard_deviation


def _get_excitatory_taus_in_range(min_tau, max_tau, size):
    mean, standard_deviation = _get_mean_and_standard_deviation_for_range(
        min_tau, max_tau)
    return np.random.normal(
        mean, standard_deviation, size)


def _get_inhibitory_taus_in_range(min_tau, max_tau, min_delta, max_delta, size):
    excitatory_mean, excitatory_standard_deviation = _get_mean_and_standard_deviation_for_range(
        min_tau, max_tau)
    delta_mean, delta_standard_deviation = _get_mean_and_standard_deviation_for_range(
        min_delta, max_delta)
    mean = excitatory_mean - delta_mean
    standard_deviation = excitatory_standard_deviation - delta_standard_deviation
    return np.random.normal(
        mean, standard_deviation, size)


def _is_odd(number):
    return number % 2 != 0


def generate_taus(hidden_neuron_count):
    if _is_odd(hidden_neuron_count):
        raise ValueError(
            f'hidden_neuron_count must be an even number but was odd. Actual value: {hidden_neuron_count}')

    excitatory_taus = _get_excitatory_taus_in_range(
        min_tau=MIN_TAU,
        max_tau=MAX_TAU,
        size=hidden_neuron_count)

    inhibitory_taus = _get_inhibitory_taus_in_range(
        min_tau=MIN_TAU,
        max_tau=MAX_TAU,
        min_delta=MIN_DELTA_TAU,
        max_delta=MAX_DELTA_TAU,
        size=hidden_neuron_count)

    return excitatory_taus, inhibitory_taus


def generate_concatenated_taus_for_input_pair(hidden_neuron_count):
    excitatory_taus, inhibitory_taus = generate_taus(hidden_neuron_count)
    half_point = len(excitatory_taus) // 2
    first_excitatory_half = excitatory_taus[:half_point]
    second_excitatory_half = excitatory_taus[half_point:]
    first_inhibitory_taus_half = inhibitory_taus[:half_point]
    second_inhibitory_taus_half = inhibitory_taus[half_point:]
    return np.concatenate([first_excitatory_half, first_inhibitory_taus_half, second_inhibitory_taus_half, second_excitatory_half])


def generate_concatenated_taus(input_neuron_count, hidden_neuron_count):
    if _is_odd(input_neuron_count):
        raise ValueError(
            f'input_neuron_count must be an even number but was odd. Actual value: {input_neuron_count}')
    input_pair_count = input_neuron_count // 2
    concatenated_taus = [generate_concatenated_taus_for_input_pair(
        _) for _ in [hidden_neuron_count] * input_pair_count]
    return reduce(np.concatenate, concatenated_taus, np.array([]))
