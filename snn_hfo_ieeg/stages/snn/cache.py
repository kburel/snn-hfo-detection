from typing import NamedTuple, Optional
from brian2 import Network, SpikeMonitor
from brian2.input.spikegeneratorgroup import SpikeGeneratorGroup
from snn_hfo_ieeg.stages.snn.model_paths import ModelPaths, load_model_paths
from snn_hfo_ieeg.stages.snn.concatenation import NeuronCount
from snn_hfo_ieeg.stages.snn.basic_network_creation import create_hidden_to_output_synapses, create_non_input_layer, create_input_layer, create_input_to_hidden_synapses
from snn_hfo_ieeg.user_facing_data import MeasurementMode
from snn_hfo_ieeg.stages.snn.artifact_filter import add_artifact_filter_to_network_and_get_interneuron, should_add_artifact_filter
from snn_hfo_ieeg.stages.snn.advanced_artifact_filter import should_add_advanced_artifact_filter, add_advanced_artifact_filter_to_network


class SpikeMonitors(NamedTuple):
    hidden: SpikeMonitor
    output: SpikeMonitor


class Cache(NamedTuple):
    input_layer: SpikeGeneratorGroup
    model_paths: ModelPaths
    neuron_counts: NeuronCount
    spike_monitors: SpikeMonitors
    network: Network
    advanced_artifact_filter_input_layer: Optional[SpikeGeneratorGroup]


def _read_neuron_counts(configuration):
    input_count = _measurement_mode_to_input_count(
        configuration.measurement_mode)
    return NeuronCount(input_count, configuration.hidden_neuron_count)


def _measurement_mode_to_input_count(measurement_mode):
    if measurement_mode is MeasurementMode.IEEG:
        return 4
    if measurement_mode is MeasurementMode.ECOG:
        return 2
    if measurement_mode is MeasurementMode.SCALP:
        return 2
    raise ValueError(
        f'measurement_mode is outside valid range. Allowed values: {MeasurementMode}, instead got: {measurement_mode}')


def _get_layers_connected_to_output_count(configuration):
    return 1 + int(should_add_artifact_filter(configuration)) + int(should_add_advanced_artifact_filter(configuration))


def create_cache(configuration):
    model_paths = load_model_paths()
    neuron_counts = _read_neuron_counts(configuration)
    hidden_layer = create_non_input_layer(
        model_paths, neuron_counts.hidden, 'hidden')
    number_of_output_neurons = 1
    layers_connected_to_output = _get_layers_connected_to_output_count(
        configuration)
    output_layer = create_non_input_layer(
        model_paths, number_of_output_neurons, 'output', num_inputs=layers_connected_to_output)
    hidden_to_output_synapses = create_hidden_to_output_synapses(
        'main', hidden_layer, output_layer, model_paths, neuron_counts)

    input_layer = create_input_layer('main', neuron_counts.input)

    input_to_hidden_synapses = create_input_to_hidden_synapses(
        name='main',
        input_layer=input_layer,
        hidden_layer=hidden_layer,
        model_paths=model_paths,
        neuron_counts=neuron_counts)

    spike_monitors = SpikeMonitors(
        hidden=SpikeMonitor(hidden_layer),
        output=SpikeMonitor(output_layer))
    network = Network(
        input_layer,
        input_to_hidden_synapses,
        hidden_layer,
        spike_monitors.hidden,
        spike_monitors.output,
        output_layer,
        hidden_to_output_synapses)

    if should_add_artifact_filter(configuration):
        add_artifact_filter_to_network_and_get_interneuron(
            model_paths, input_layer, output_layer, network)

    advanced_artifact_filter_input_layer = add_advanced_artifact_filter_to_network(
        network, output_layer, model_paths, neuron_counts) if should_add_advanced_artifact_filter(configuration) else None

    network.store()

    return Cache(
        model_paths=model_paths,
        neuron_counts=neuron_counts,
        spike_monitors=spike_monitors,
        network=network,
        input_layer=input_layer,
        advanced_artifact_filter_input_layer=advanced_artifact_filter_input_layer
    )
