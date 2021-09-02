import numpy as np
from snn_hfo_ieeg.stages.snn.basic_network_creation import create_synapses, create_non_input_layer, create_hidden_to_output_synapses, create_input_layer, create_input_to_hidden_synapses
from snn_hfo_ieeg.user_facing_data import MeasurementMode

NAME = 'advanced_artifact_filter'


def should_add_advanced_artifact_filter(configuration):
    return configuration.measurement_mode is MeasurementMode.SCALP


def create_advanced_artifact_filter_output_to_output_synapses(advanced_artifact_filter_output, output_layer, model_paths):
    weights = np.array([-10_000])
    taus = np.array([10])
    return create_synapses(
        f'{NAME}_output_to_output', model_paths, advanced_artifact_filter_output, output_layer, weights, taus)


def get_advanced_artifact_filter_input_bandwidth(filtered_spikes):
    return [filtered_spikes.above_fast_ripple]


def add_input_to_advanced_artifact_filter_to_network(cache, filtered_spikes):
    advanced_artifact_filter_filtered_bandwidths = get_advanced_artifact_filter_input_bandwidth(
        filtered_spikes)
    advanced_artifact_filter_input_layer = create_input_layer(
        NAME,
        advanced_artifact_filter_filtered_bandwidths,
        cache.neuron_counts.input)

    advanced_artifact_filter_input_to_hidden_synapses = create_input_to_hidden_synapses(
        name=NAME,
        input_layer=advanced_artifact_filter_input_layer,
        hidden_layer=cache.advanced_artifact_filter_hidden_layer,
        cache=cache)
    cache.network.add(advanced_artifact_filter_input_layer)
    cache.network.add(advanced_artifact_filter_input_to_hidden_synapses)
    return advanced_artifact_filter_input_layer, advanced_artifact_filter_input_to_hidden_synapses


def add_advanced_artifact_filter_to_network(network, output_layer, model_paths, neuron_counts):
    hidden_layer = create_non_input_layer(
        model_paths, neuron_counts.hidden, f'{NAME}_hidden')
    number_of_output_neurons = 1
    advanced_artifact_filter_output_layer = create_non_input_layer(
        model_paths, number_of_output_neurons, '{NAME}_output', num_inputs=2)
    hidden_to_output_synapses = create_hidden_to_output_synapses(
        NAME, hidden_layer, advanced_artifact_filter_output_layer, model_paths, neuron_counts)
    advanced_artifact_filter_output_to_output_synapses = create_advanced_artifact_filter_output_to_output_synapses(
        advanced_artifact_filter_output_layer, output_layer, model_paths)
    network.add(
        hidden_layer,
        advanced_artifact_filter_output_layer,
        hidden_to_output_synapses,
        advanced_artifact_filter_output_to_output_synapses)
    return hidden_layer
