from typing import List
from functools import reduce
import warnings
from brian2.units import second
from snn_hfo_ieeg.user_facing_data import Bandwidth, FilteredSpikes, MeasurementMode
from snn_hfo_ieeg.stages.snn.cache import Cache, SpikeMonitors, create_cache
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes


def _get_relevant_input_bandwidth(measurement_mode, filtered_spikes: FilteredSpikes) -> List[Bandwidth]:
    if measurement_mode is MeasurementMode.IEEG:
        return [filtered_spikes.ripple, filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.ECOG:
        return [filtered_spikes.fast_ripple]
    if measurement_mode is MeasurementMode.SCALP:
        return [filtered_spikes.ripple]
    raise ValueError(
        f'configuration.measurement_mode has an invalid value. Allowed values: {MeasurementMode}, instead got: {measurement_mode}')


def _append_spikes(spikes, spike_train):
    spikes.append(spike_train.up)
    spikes.append(spike_train.down)
    return spikes


def _concatenate_bandwidths(bandwidths):
    spike_trains = [
        bandwidth.spike_trains for bandwidth in bandwidths if bandwidth is not None]
    spikes = reduce(_append_spikes, spike_trains, [])
    return concatenate_spikes(spikes)


def snn_stage(filtered_spikes, duration, configuration, cache: Cache) -> SpikeMonitors:
    warnings.simplefilter("ignore", DeprecationWarning)
    if cache is None:
        cache = create_cache(configuration)

    cache.network.restore()

    bandwidths = _get_relevant_input_bandwidth(
        configuration.measurement_mode, filtered_spikes)
    input_spiketimes, input_neurons_id = _concatenate_bandwidths(bandwidths)

    cache.input_layer.set_spikes(input_neurons_id, input_spiketimes * second)

    cache.network.run(duration * second)

    return cache.spike_monitors
