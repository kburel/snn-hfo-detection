import numpy as np

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


def generate_taus(hidden_neuron_count):
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

