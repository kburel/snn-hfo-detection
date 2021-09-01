from functools import reduce
from typing import NamedTuple, Optional
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
from snn_hfo_ieeg.user_facing_data import MeasurementMode
from snn_hfo_ieeg.stages.snn.artifact_filter import add_artifact_filter_to_network_and_get_interneuron, add_input_to_artifact_filter_to_network
from snn_hfo_ieeg.stages.snn.creation import create_non_input_layer, create_synapses


class SpikeMonitors(NamedTuple):
    hidden: SpikeMonitor
    output: SpikeMonitor


def _append_spikes(spikes, spike_train):
    spikes.append(spike_train.up)
    spikes.append(spike_train.down)
    return spikes


def _concatenate_filtered_spikes(filtered_spikes):
    spike_trains = [
        bandwidth.spike_trains for bandwidth in filtered_spikes if bandwidth is not None]
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


def _create_input_layer(filtered_spikes, input_count):
    input_spiketimes, input_neurons_id = _concatenate_filtered_spikes(
        filtered_spikes)
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
    network: Network


def _should_add_artifact_filter(configuration):
    return configuration.measurement_mode is MeasurementMode.ECOG or configuration.measurement_mode is MeasurementMode.SCALP


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
        model_paths, output_layer, network) if _should_add_artifact_filter(configuration) else None

    network.store()

    return Cache(
        model_paths=model_paths,
        neuron_counts=neuron_counts,
        hidden_layer=hidden_layer,
        spike_monitors=spike_monitors,
        interneuron=interneuron,
        network=network
    )


def snn_stage(filtered_spikes, duration, configuration, cache: Cache) -> SpikeMonitors:
    warnings.simplefilter("ignore", DeprecationWarning)
    if cache is None:
        cache = _create_cache(configuration)

    cache.network.restore()

    input_layer = _create_input_layer(
        filtered_spikes, cache.neuron_counts.input)
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

    cache.network.run(duration * second)

    cache.network.remove(input_layer)
    cache.network.remove(input_to_hidden_synapses)

    if cache.interneuron is not None:
        cache.network.remove(input_to_interneuron_synapses)

    return cache.spike_monitors
