import numpy as np
from snn_hfo_ieeg.stages.snn.creation import create_synapses
from snn_hfo_ieeg.user_facing_data import MeasurementMode


def should_add_signal_enhancer(configuration):
    return configuration.measurement_mode is MeasurementMode.SCALP


def create_signal_enhancer_to_output_synapses(signal_enhancer_layer, output_layer, model_paths, hidden_neuron_count):
    weights = np.repeat(3_000.0, hidden_neuron_count.hidden)
    taus = np.repeat(10, hidden_neuron_count.hidden)
    return create_synapses(
        'signal_enhancer_to_output', model_paths, signal_enhancer_layer, output_layer, weights, taus)


def get_signal_enhancer_input_bandwidth(filtered_spikes):
    return [filtered_spikes.ripple, filtered_spikes.above_fast_ripple]
