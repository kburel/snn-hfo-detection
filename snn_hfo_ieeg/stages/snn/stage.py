from typing import List
import warnings
from brian2.units import second
from snn_hfo_ieeg.stages.snn.basic_network import create_input_layer, create_input_to_hidden_synapses
from snn_hfo_ieeg.user_facing_data import Bandwidth, FilteredSpikes, MeasurementMode
from snn_hfo_ieeg.stages.snn.artifact_filter import add_input_to_artifact_filter_to_network
from snn_hfo_ieeg.stages.snn.cache import Cache, SpikeMonitors, create_cache
from snn_hfo_ieeg.stages.snn.signal_enhancer import add_input_to_signal_enhancer_to_network


def _get_relevant_input_bandwidth(measurement_mode, filtered_spikes: FilteredSpikes) -> List[Bandwidth]:
    if measurement_mode is MeasurementMode.IEEG:
        return [filtered_spikes.ripple, filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.ECOG:
        return [filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.SCALP:
        return [filtered_spikes.ripple]
    raise ValueError(
        f'configuration.measurement_mode has an invalid value. Allowed values: {MeasurementMode}, instead got: {measurement_mode}')


def snn_stage(filtered_spikes, duration, configuration, cache: Cache) -> SpikeMonitors:
    warnings.simplefilter("ignore", DeprecationWarning)
    if cache is None:
        cache = create_cache(configuration)

    cache.network.restore()

    input_filtered_bandwidths = _get_relevant_input_bandwidth(
        configuration.measurement_mode,
        filtered_spikes)
    input_layer = create_input_layer(
        'main',
        input_filtered_bandwidths,
        cache.neuron_counts.input)

    input_to_hidden_synapses = create_input_to_hidden_synapses(
        name='main',
        input_layer=input_layer,
        hidden_layer=cache.hidden_layer,
        cache=cache)

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
