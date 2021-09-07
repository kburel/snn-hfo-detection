from typing import List
from functools import reduce
from brian2.units import second
from snn_hfo_detection.user_facing_data import Bandwidth, FilteredSpikes, MeasurementMode
from snn_hfo_detection.functions.signal_to_spike.utility import concatenate_spikes
from snn_hfo_detection.stages.snn.advanced_artifact_filter import get_advanced_artifact_filter_input_bandwidth


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


def _set_input_layer_to_bandwidths(input_layer, bandwidths):
    input_spiketimes, input_neurons_id = _concatenate_bandwidths(bandwidths)
    input_layer.set_spikes(
        input_neurons_id, input_spiketimes * second, sorted=True)


def set_input_spikes(filtered_spikes, input_layer, measurement_mode):
    bandwidths = _get_relevant_input_bandwidth(
        measurement_mode, filtered_spikes)
    _set_input_layer_to_bandwidths(input_layer, bandwidths)


def set_advanced_artifact_filter_input_spikes(filtered_spikes, advanced_artifact_filter_input_layer):
    bandwidths = get_advanced_artifact_filter_input_bandwidth(filtered_spikes)
    _set_input_layer_to_bandwidths(
        advanced_artifact_filter_input_layer, bandwidths)
