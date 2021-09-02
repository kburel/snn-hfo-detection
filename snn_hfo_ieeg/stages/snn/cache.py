from typing import NamedTuple, Optional
from brian2 import Network, SpikeMonitor
from teili.core.groups import Neurons
from snn_hfo_ieeg.stages.snn.model_paths import ModelPaths, load_model_paths
from snn_hfo_ieeg.stages.snn.concatenation import NeuronCount
from snn_hfo_ieeg.stages.snn.basic_network import create_hidden_to_output_synapses
from snn_hfo_ieeg.user_facing_data import MeasurementMode
from snn_hfo_ieeg.stages.snn.artifact_filter import add_artifact_filter_to_network_and_get_interneuron, should_add_artifact_filter
from snn_hfo_ieeg.stages.snn.creation import create_non_input_layer
from snn_hfo_ieeg.stages.snn.signal_enhancer import should_add_signal_enhancer, add_signal_enhancer_to_network


class SpikeMonitors(NamedTuple):
    hidden: SpikeMonitor
    output: SpikeMonitor


class Cache(NamedTuple):
    model_paths: ModelPaths
    neuron_counts: NeuronCount
    hidden_layer: Neurons
    spike_monitors: SpikeMonitors
    interneuron: Optional[Neurons]
    signal_enhancer_hidden_layer: Optional[Neurons]
    network: Network


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
    if configuration.measurement_mode is MeasurementMode.IEEG:
        return 1
    if configuration.measurement_mode is MeasurementMode.ECOG:
        return 2
    if configuration.measurement_mode is MeasurementMode.SCALP:
        return 3
    raise ValueError(
        f'measurement_mode is outside valid range. Allowed values: {MeasurementMode}, instead got: {configuration.measurement_mode}')


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
        hidden_layer, output_layer, model_paths, neuron_counts)

    spike_monitors = SpikeMonitors(
        hidden=SpikeMonitor(hidden_layer),
        output=SpikeMonitor(output_layer))
    network = Network(
        hidden_layer,
        spike_monitors.hidden,
        spike_monitors.output,
        output_layer,
        hidden_to_output_synapses)

    interneuron = add_artifact_filter_to_network_and_get_interneuron(
        model_paths, output_layer, network) if should_add_artifact_filter(configuration) else None

    signal_enhancer_hidden_layer = add_signal_enhancer_to_network(
        network, output_layer, model_paths, neuron_counts) if should_add_signal_enhancer(configuration) else None

    network.store()

    return Cache(
        model_paths=model_paths,
        neuron_counts=neuron_counts,
        hidden_layer=hidden_layer,
        spike_monitors=spike_monitors,
        interneuron=interneuron,
        network=network,
        signal_enhancer_hidden_layer=signal_enhancer_hidden_layer
    )
