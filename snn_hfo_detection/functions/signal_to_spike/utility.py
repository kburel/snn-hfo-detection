from typing import NamedTuple, Optional
import numpy as np


class SignalToSpikeParameters(NamedTuple):
    signal: np.ndarray
    threshold_up: float
    threshold_down: float
    times: np.ndarray
    refractory_period: float
    interpolation_factor: Optional[float]


class SpikeTrains(NamedTuple):
    '''
    Up and down spike trains received by filtering a signal
    '''
    up: np.array
    down: np.array


def find_thresholds(signals, times, window_size, sample_ratio, scaling_factor):
    '''
    This functions retuns the mean threshold for your signals, based on the calculated
    mean noise floor and a user-specified scaling facotr that depeneds on the type of signals,
    characteristics of patterns, etc.

    Parameters
    -------
    signals : array
        amplitude of the signals
    times : array
        time vector
    window : float
        time window [same units as time vector] where the maximum amplitude
        of the signals will be calculated
    sample_ratio : float
        the percentage of time windows that will be used to
        calculate the mean maximum amplitude.
     scaling_factor : float
        a percentage of the calculated threshold
    '''
    min_time = np.min(times)
    if np.min(times) < 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a negative time: {min_time}')
    duration = np.max(times) - min_time
    if duration <= 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a duration that under or equal to zero. Got duration: {duration}')

    if len(signals) == 0:
        raise ValueError('signals is not allowed to be empty, but was'
                         )
    if len(times) == 0:
        raise ValueError('times is not allowed to be empty, but was')

    if len(signals) != len(times):
        raise ValueError(
            f'signals and times need to have corresponding indices, but signals has length {len(signals)} while times has length {len(times)}')

    if not 0 < sample_ratio < 1:
        raise ValueError(
            f'sample_ratio must be a value between 0 and 1, but was {sample_ratio}'
        )

    num_timesteps = int(np.ceil(duration / window_size))
    max_min_amplitude = np.zeros((num_timesteps, 2))
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=duration, step=window_size)):
        interval_end = interval_start + window_size
        index = np.where((times >= interval_start) & (times <= interval_end))
        max_amplitude = np.max(signals[index])
        min_amplitude = np.min(signals[index])
        max_min_amplitude[interval_nr, 0] = max_amplitude
        max_min_amplitude[interval_nr, 1] = min_amplitude

    chosen_samples = max(int(np.round(num_timesteps * sample_ratio)), 1)
    threshold_up = np.mean(np.sort(max_min_amplitude[:, 0])[:chosen_samples])
    threshold_dn = np.mean(
        np.sort(max_min_amplitude[:, 1] * -1)[:chosen_samples])
    return scaling_factor*(threshold_up + threshold_dn)


# ========================================================================================
# List of spiketimes for the SNN input
# ========================================================================================
def concatenate_spikes(spikes):
    '''
    Get spikes per channel in a dictionary and concatenate them in one single vector with
    spike times and neuron ids.
    :param spikes_list (list): list of sampled frequencies, where every frequency is a list of spike times
    :return all_spiketimes (array): vector of all spike times
    :return all_neuron_ids (array): vector of all neuron ids
    '''
    all_spiketimes = []
    all_neuron_ids = []
    channel_nr = 0
    for spike in spikes:
        if channel_nr == 0:
            all_spiketimes = spike
            all_neuron_ids = np.ones_like(all_spiketimes) * channel_nr
            channel_nr += 1
        else:
            new_spiketimes = spike
            all_spiketimes = np.concatenate(
                (all_spiketimes, new_spiketimes), axis=0)
            all_neuron_ids = np.concatenate((all_neuron_ids,
                                             np.ones_like(new_spiketimes) * channel_nr), axis=0)
            channel_nr += 1

    sorted_index = np.argsort(all_spiketimes)
    all_spiketimes_new = all_spiketimes[sorted_index]
    all_neuron_ids_new = all_neuron_ids[sorted_index]
    return all_spiketimes_new, all_neuron_ids_new


def get_sampling_frequency(times) -> float:
    return 1 / (times[1] - times[0])
