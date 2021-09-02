import numpy as np
from snn_hfo_ieeg.stages.snn.creation import create_synapses, create_non_input_layer
from snn_hfo_ieeg.stages.snn.basic_network import create_input_layer, create_input_to_hidden_synapses
from snn_hfo_ieeg.user_facing_data import MeasurementMode


def should_add_signal_enhancer(configuration):
    return configuration.measurement_mode is MeasurementMode.SCALP


def create_signal_enhancer_hidden_to_output_synapses(hidden_layer, output_layer, model_paths, hidden_neuron_count):
    weights = np.repeat(3_000.0, hidden_neuron_count.hidden)
    taus = np.repeat(10, hidden_neuron_count.hidden)
    return create_synapses(
        'signal_enhancer_hidden_to_output', model_paths, hidden_layer, output_layer, weights, taus)


def create_signal_enhancer_output_to_output_synapses(signal_enhancer_output, output_layer, model_paths):
    weights = np.array([3_000])
    taus = np.array([5])
    return create_synapses(
        'signal_enhancer_output_to_output', model_paths, signal_enhancer_output, output_layer, weights, taus)


def get_signal_enhancer_input_bandwidth(filtered_spikes):
    return [filtered_spikes.above_fast_ripple]


def add_input_to_signal_enhancer_to_network(cache, filtered_spikes):
    signal_enhancer_filtered_bandwidths = get_signal_enhancer_input_bandwidth(
        filtered_spikes)
    signal_enhancer_input_layer = create_input_layer(
        'signal_enhancer',
        signal_enhancer_filtered_bandwidths,
        cache.neuron_counts.input)

    signal_enhancer_input_to_hidden_synapses = create_input_to_hidden_synapses(
        name='signal_enhancer',
        input_layer=signal_enhancer_input_layer,
        hidden_layer=cache.signal_enhancer_hidden_layer,
        cache=cache)
    cache.network.add(signal_enhancer_input_layer)
    cache.network.add(signal_enhancer_input_to_hidden_synapses)
    return signal_enhancer_input_layer, signal_enhancer_input_to_hidden_synapses


def add_signal_enhancer_to_network(network, output_layer, model_paths, neuron_counts):
    hidden_layer = create_non_input_layer(
        model_paths, neuron_counts.hidden, 'signal_enhancer_hidden')
    number_of_output_neurons = 1
    signal_enhancer_output_layer = create_non_input_layer(
        model_paths, number_of_output_neurons, 'signal_enhancer_output', num_inputs=2)
    hidden_to_output_synapses = create_signal_enhancer_hidden_to_output_synapses(
        hidden_layer, signal_enhancer_output_layer, model_paths, neuron_counts)
    signal_enhancer_output_to_output_synapses = create_signal_enhancer_output_to_output_synapses(
        signal_enhancer_output_layer, output_layer, model_paths)
    network.add(
        hidden_layer,
        signal_enhancer_output_layer,
        hidden_to_output_synapses,
        signal_enhancer_output_to_output_synapses)
    return hidden_layer
