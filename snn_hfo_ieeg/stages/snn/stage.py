from functools import reduce
from typing import NamedTuple, Optional, List
import warnings
from brian2 import Network, SpikeGeneratorGroup, SpikeMonitor
from brian2.units import us, second
import numpy as np
from teili.core.groups import Neurons
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes
from snn_hfo_ieeg.stages.snn.tau_generation import generate_concatenated_taus
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights
from snn_hfo_ieeg.stages.snn.model_paths import ModelPaths, load_model_paths
from snn_hfo_ieeg.stages.snn.concatenation import NeuronCount
from snn_hfo_ieeg.user_facing_data import Bandwidth, FilteredSpikes, MeasurementMode
from snn_hfo_ieeg.stages.snn.artifact_filter import add_artifact_filter_to_network_and_get_interneuron, add_input_to_artifact_filter_to_network, should_add_artifact_filter
from snn_hfo_ieeg.stages.snn.creation import create_non_input_layer, create_synapses
from snn_hfo_ieeg.stages.snn.signal_enhancer import create_signal_enhancer_to_output_synapses, get_signal_enhancer_input_bandwidth, should_add_signal_enhancer


class SpikeMonitors(NamedTuple):
    hidden: SpikeMonitor
    output: SpikeMonitor


def _append_spikes(spikes, spike_train):
    spikes.append(spike_train.up)
    spikes.append(spike_train.down)
    return spikes


def _get_relevant_input_bandwidth(measurement_mode, filtered_spikes: FilteredSpikes) -> List[Bandwidth]:
    if measurement_mode is MeasurementMode.IEEG:
        return [filtered_spikes.ripple, filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.ECOG:
        return [filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.SCALP:
        return [filtered_spikes.ripple]
    raise ValueError(
        f'configuration.measurement_mode has an invalid value. Allowed values: {MeasurementMode}, instead got: {measurement_mode}')


def _concatenate_filtered_spikes(bandwidths):
    spike_trains = [
        bandwidth.spike_trains for bandwidth in bandwidths if bandwidth is not None]
    spikes = reduce(_append_spikes, spike_trains, [])
    return concatenate_spikes(spikes)


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


def _create_input_layer(bandwidths, input_count):
    input_spiketimes, input_neurons_id = _concatenate_filtered_spikes(
        bandwidths)
    return SpikeGeneratorGroup(input_count,
                               input_neurons_id,
                               input_spiketimes*second,
                               dt=100*us, name='input')


def _create_input_to_hidden_synapses(input_layer, hidden_layer, model_paths, neuron_counts):
    weights = generate_weights(neuron_counts)
    taus = generate_concatenated_taus(neuron_counts)
    return create_synapses(
        'input_to_hidden', model_paths, input_layer, hidden_layer, weights, taus)


def _create_hidden_to_output_synapses(hidden_layer, output_layer, model_paths, hidden_neuron_count):
    weights = np.repeat(3_000.0, hidden_neuron_count.hidden)
    taus = np.repeat(10, hidden_neuron_count.hidden)
    return create_synapses(
        'hidden_to_output', model_paths, hidden_layer, output_layer, weights, taus)


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
    hidden_to_output_synapses = _create_hidden_to_output_synapses(
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
    input_layer = _create_input_layer(
        input_filtered_bandwidths, cache.neuron_counts.input)

    input_to_hidden_synapses = _create_input_to_hidden_synapses(
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
        signal_enhancer_filtered_bandwidths = get_signal_enhancer_input_bandwidth(
            filtered_spikes)
        signal_enhancer_input_layer = _create_input_layer(
            signal_enhancer_filtered_bandwidths, cache.neuron_counts.input)

        signal_enhancer_input_to_hidden_synapses = _create_input_to_hidden_synapses(
            signal_enhancer_input_layer,
            cache.signal_enhancer_hidden_layer,
            cache.model_paths,
            cache.neuron_counts)
        cache.network.add(signal_enhancer_input_layer)
        cache.network.add(signal_enhancer_input_to_hidden_synapses)

    cache.network.run(duration * second)

    cache.network.remove(input_layer)
    cache.network.remove(input_to_hidden_synapses)

    if cache.interneuron is not None:
        cache.network.remove(input_to_interneuron_synapses)

    if cache.signal_enhancer_hidden_layer is not None:
        cache.network.remove(signal_enhancer_input_layer)
        cache.network.remove(signal_enhancer_input_to_hidden_synapses)

    return cache.spike_monitors
