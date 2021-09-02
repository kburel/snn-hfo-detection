from typing import NamedTuple, Optional, List
import warnings
from brian2 import Network, SpikeMonitor
from brian2.units import second
from teili.core.groups import Neurons
from snn_hfo_ieeg.stages.snn.model_paths import ModelPaths, load_model_paths
from snn_hfo_ieeg.stages.snn.concatenation import NeuronCount
from snn_hfo_ieeg.stages.snn.basic_network import create_input_layer, create_input_to_hidden_synapses, create_hidden_to_output_synapses
from snn_hfo_ieeg.user_facing_data import Bandwidth, FilteredSpikes, MeasurementMode
from snn_hfo_ieeg.stages.snn.artifact_filter import add_artifact_filter_to_network_and_get_interneuron, add_input_to_artifact_filter_to_network, should_add_artifact_filter
from snn_hfo_ieeg.stages.snn.creation import create_non_input_layer
from snn_hfo_ieeg.stages.snn.signal_enhancer import create_signal_enhancer_to_output_synapses, should_add_signal_enhancer, add_input_to_signal_enhancer_to_network


class SpikeMonitors(NamedTuple):
    hidden: SpikeMonitor
    output: SpikeMonitor


def _get_relevant_input_bandwidth(measurement_mode, filtered_spikes: FilteredSpikes) -> List[Bandwidth]:
    if measurement_mode is MeasurementMode.IEEG:
        return [filtered_spikes.ripple, filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.ECOG:
        return [filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.SCALP:
        return [filtered_spikes.ripple]
    raise ValueError(
        f'configuration.measurement_mode has an invalid value. Allowed values: {MeasurementMode}, instead got: {measurement_mode}')


def _measurement_mode_to_input_count(measurement_mode):
    if measurement_mode is MeasurementMode.IEEG:
        return 4
    if measurement_mode is MeasurementMode.ECOG:
        return 2
    if measurement_mode is MeasurementMode.SCALP:
        return 2
    raise ValueError(
        f'measurement_mode is outside valid range. Allowed values: {MeasurementMode}, instead got: {measurement_mode}')


def _read_neuron_counts(configuration):
    input_count = _measurement_mode_to_input_count(
        configuration.measurement_mode)
    return NeuronCount(input_count, configuration.hidden_neuron_count)


class Cache(NamedTuple):
    model_paths: ModelPaths
    neuron_counts: NeuronCount
    hidden_layer: Neurons
    spike_monitors: SpikeMonitors
    interneuron: Optional[Neurons]
    signal_enhancer_hidden_layer: Optional[Neurons]
    network: Network


def _add_signal_enhancer_to_network(network, output_layer, model_paths, neuron_counts):
    hidden_layer = create_non_input_layer(
        model_paths, neuron_counts.hidden, 'signal_enhancer_hidden')
    number_of_output_neurons = 1
    signal_enhancer_output_layer = create_non_input_layer(
        model_paths, number_of_output_neurons, 'signal_enhancer_output', num_inputs=2)
    signal_enhancer_to_output_synapses = create_signal_enhancer_to_output_synapses(
        signal_enhancer_output_layer, output_layer, model_paths, neuron_counts.hidden)
    network.add(
        hidden_layer,
        signal_enhancer_output_layer,
        signal_enhancer_to_output_synapses)
    return hidden_layer


def _create_cache(configuration):
    model_paths = load_model_paths()
    neuron_counts = _read_neuron_counts(configuration)
    hidden_layer = create_non_input_layer(
        model_paths, neuron_counts.hidden, 'hidden')
    number_of_output_neurons = 1
    output_layer = create_non_input_layer(
        model_paths, number_of_output_neurons, 'output', num_inputs=2)
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

    signal_enhancer_hidden_layer = _add_signal_enhancer_to_network if should_add_signal_enhancer(
        configuration) else None

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


def snn_stage(filtered_spikes, duration, configuration, cache: Cache) -> SpikeMonitors:
    warnings.simplefilter("ignore", DeprecationWarning)
    if cache is None:
        cache = _create_cache(configuration)

    cache.network.restore()

    input_filtered_bandwidths = _get_relevant_input_bandwidth(
        configuration.measurement_mode, filtered_spikes)
    input_layer = create_input_layer(
        input_filtered_bandwidths, cache.neuron_counts.input)

    input_to_hidden_synapses = create_input_to_hidden_synapses(
        input_layer,
        cache.hidden_layer,
        cache.model_paths,
        cache.neuron_counts)

    cache.network.add(input_layer)
    cache.network.add(input_to_hidden_synapses)

    if cache.interneuron is not None:
        input_to_interneuron_synapses = add_input_to_artifact_filter_to_network(
            input_layer, cache)

    if cache.signal_enhancer_hidden_layer is not None:
        signal_enhancer_input_layer, signal_enhancer_input_to_hidden_synapses = add_input_to_signal_enhancer_to_network(
            cache, filtered_spikes)

    cache.network.run(duration * second)

    cache.network.remove(input_layer)
    cache.network.remove(input_to_hidden_synapses)

    if cache.interneuron is not None:
        cache.network.remove(input_to_interneuron_synapses)

    if cache.signal_enhancer_hidden_layer is not None:
        cache.network.remove(signal_enhancer_input_layer)
        cache.network.remove(signal_enhancer_input_to_hidden_synapses)

    return cache.spike_monitors
