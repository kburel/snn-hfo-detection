import numpy as np
from snn_hfo_ieeg.stages.snn.concatenation import concatenate_excitatory_and_inhibitory_with_generator_function

MIN_TAU = 3
MAX_TAU = 6
MIN_DELTA_TAU = 0.3
MAX_DELTA_TAU = 1

OUTLIER_FRACTION = 0.02


def _get_mean_and_standard_deviation_for_range(min, max):
    mean = np.mean([min, max])
    # 98 % (100% - OUTLIER_FRACTION) of all data lies within 2.3 standard deviations on a normal distribution
    standard_deviation = (max - mean) / 2.4
    return mean, standard_deviation


def _get_abs_normal_range(min, max, size):
    mean, standard_deviation = _get_mean_and_standard_deviation_for_range(
        min, max)
    normal_range = np.random.normal(mean, standard_deviation, size)
    return abs(normal_range)


def generate_taus(hidden_neuron_count):
    excitatory_taus = _get_abs_normal_range(
        min=MIN_TAU,
        max=MAX_TAU,
        size=hidden_neuron_count)
    deltas = _get_abs_normal_range(
        min=MIN_DELTA_TAU,
        max=MAX_DELTA_TAU,
        size=hidden_neuron_count)

    inhibitory_taus = excitatory_taus - deltas

    return excitatory_taus, inhibitory_taus


def generate_concatenated_taus(neuron_count):
    return concatenate_excitatory_and_inhibitory_with_generator_function(neuron_count, generate_taus)
