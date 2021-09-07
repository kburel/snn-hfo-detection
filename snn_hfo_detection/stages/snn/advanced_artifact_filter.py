import numpy as np
from snn_hfo_detection.stages.snn.basic_network_creation import create_synapses, create_non_input_layer, create_input_layer
from snn_hfo_detection.user_facing_data import MeasurementMode
from snn_hfo_detection.stages.snn.tau_generation import generate_concatenated_taus
from snn_hfo_detection.stages.snn.weight_generation import generate_weights

NAME = 'advanced_artifact_filter'


def should_add_advanced_artifact_filter(configuration):
    return configuration.measurement_mode is MeasurementMode.SCALP


def create_advanced_artifact_filter_output_to_output_synapses(advanced_artifact_filter_output, output_layer, model_paths):
    weights = np.array([-100_000])
    taus = np.array([100])
    return create_synapses(
        f'{NAME}_output_to_output', model_paths, advanced_artifact_filter_output, output_layer, weights, taus)


def get_advanced_artifact_filter_input_bandwidth(filtered_spikes):
    return [filtered_spikes.above_fast_ripple]


def _create_input_to_hidden_synapses(name, input_layer, hidden_layer, model_paths, neuron_counts):
    weights = generate_weights(neuron_counts) * 2.0
    taus = generate_concatenated_taus(neuron_counts)
    return create_synapses(
        f'{name}_input_to_hidden', model_paths, input_layer, hidden_layer, weights, taus)


def _create_hidden_to_output_synapses(name, hidden_layer, output_layer, model_paths, neuron_counts):
    weights = np.repeat(20_000.0, neuron_counts.hidden)
    taus = np.repeat(10, neuron_counts.hidden)
    return create_synapses(
        f'{name}_hidden_to_{name}_output', model_paths, hidden_layer, output_layer, weights, taus)


def add_advanced_artifact_filter_to_network(network, output_layer, model_paths, neuron_counts):
    input_layer = create_input_layer(
        NAME,
        neuron_counts.input)
    hidden_layer = create_non_input_layer(
        model_paths, neuron_counts.hidden, f'{NAME}_hidden')
    input_to_hidden_synapses = _create_input_to_hidden_synapses(
        name=NAME,
        input_layer=input_layer,
        hidden_layer=hidden_layer,
        model_paths=model_paths,
        neuron_counts=neuron_counts)
    number_of_output_neurons = 1
    advanced_artifact_filter_output_layer = create_non_input_layer(
        model_paths, number_of_output_neurons, f'{NAME}_output')
    hidden_to_output_synapses = _create_hidden_to_output_synapses(
        NAME, hidden_layer, advanced_artifact_filter_output_layer, model_paths, neuron_counts)
    advanced_artifact_filter_output_to_output_synapses = create_advanced_artifact_filter_output_to_output_synapses(
        advanced_artifact_filter_output_layer, output_layer, model_paths)

    network.add(
        input_layer,
        hidden_layer,
        advanced_artifact_filter_output_layer,
        hidden_to_output_synapses,
        advanced_artifact_filter_output_to_output_synapses,
        input_to_hidden_synapses,)
    return input_layer
