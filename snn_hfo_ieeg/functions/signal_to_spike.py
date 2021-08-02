from typing import NamedTuple
import scipy as sc
import numpy as np

# ========================================================================================
# Threshold calculation based on the noise floor
# ========================================================================================


class SpikeTrains(NamedTuple):
    '''
    Up and down spike trains received by filtering a signal
    '''
    up: np.array
    down: np.array


def find_thresholds(signals, times, window_size, step_size, sample_ratio, scaling_factor):
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
    if step_size > window_size:
        raise ValueError(
            f'step_size needs to be at most windows_size, but got: step_size={step_size}, window_size={step_size}')
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

    num_timesteps = int(np.ceil(duration / step_size))
    max_min_amplitude = np.zeros((num_timesteps, 2))
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=duration, step=step_size)):
        interval_end = interval_start + window_size
        index = np.where((times >= interval_start) & (times <= interval_end))
        max_amplitude = np.max(signals[index])
        min_amplitude = np.min(signals[index])
        max_min_amplitude[interval_nr, 0] = max_amplitude
        max_min_amplitude[interval_nr, 1] = min_amplitude

    chosen_samples = int(np.round(num_timesteps * sample_ratio))
    threshold_up = np.mean(np.sort(max_min_amplitude[:, 0])[:chosen_samples])
    threshold_dn = np.mean(
        np.sort(max_min_amplitude[:, 1] * -1)[:chosen_samples])
    return scaling_factor*(threshold_up + threshold_dn)


# ========================================================================================
# Signal to spike conversion with refractory period
# ========================================================================================
def signal_to_spike_refractory(interpfact, times, amplitude, thr_up, thr_dn, refractory_period):
    '''
    This functions retuns two spike trains, when the signal crosses the specified threshold in
    a rising direction (UP spikes) and when it crosses the specified threshold in a falling
    direction (DOWN spikes)

    :times (array): time vector
    :amplitude (array): amplitude of the signal
    :interpfact (int): upsampling factor, new sampling frequency
    :thr_up (float): threshold crossing in a rising direction
    :thr_dn (float): threshold crossing in a falling direction
    :refractory_period (float): period in which no spike will be generated [same units as time vector]
    '''
    actual_dc = 0
    spike_up = []
    spike_dn = []

    intepolated_time = sc.interpolate.interp1d(times, amplitude)
    rangeint = np.round((np.max(times) - np.min(times))*interpfact)
    xnew = np.linspace(np.min(times), np.max(
        times), num=int(rangeint), endpoint=True)
    data = np.reshape([xnew, intepolated_time(xnew)], (2, len(xnew))).T

    i = 0
    while i < (len(data)):
        if((actual_dc + thr_up) < data[i, 1]):
            spike_up.append(data[i, 0])  # spike up
            actual_dc = data[i, 1]        # update current dc value
            i += int(refractory_period * interpfact)
        elif((actual_dc - thr_dn) > data[i, 1]):
            spike_dn.append(data[i, 0])  # spike dn
            actual_dc = data[i, 1]        # update curre
            i += int(refractory_period * interpfact)
        else:
            i += 1

    return SpikeTrains(up=spike_up,
                       down=spike_dn)


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
