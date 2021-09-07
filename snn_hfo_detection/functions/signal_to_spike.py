from typing import NamedTuple
import numpy as np
from numba import njit

# ========================================================================================
# Threshold calculation based on the noise floor
# ========================================================================================


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


# Usage:
    # converts signal into UP and DOWN spikes using the Adaptive Delta Modulation scheme.
    # spike times are determined through linear interpolation between the current and the next sampling point

# Input Parameters:
    # input_signal (array): input signal
    # threshold_up (float): threshold for UP spikes to occur
    # threshold_down (float): threshold for DOWN spikes to occur
    # sampling_frequency (float): sampling frequency of the input signal (1/dt)
    # refractory_period_duration (float): refractory period in which spikes can not occur (NEEDS TO BE AT LEAST 1/sampling_frequency)

# Output Parameters:
    # spike_t_up (array): list of precise UP spike times
    # spike_t_dn (array): list of precise DOWN spike times
@njit(fastmath=True, parallel=True)
def signal_to_spike(input_signal, threshold_up, threshold_down, times, refractory_period_duration) -> SpikeTrains:
    sampling_frequency = get_sampling_frequency(times)
    if refractory_period_duration < sampling_frequency:
        raise ValueError(
            f'Refractory period ({refractory_period_duration}) is smaller than sampling frequency ({sampling_frequency})')
    delta_time = 1/sampling_frequency
    dc_voltage = input_signal[0]
    remainder_of_refractory = 0
    spike_t_up = times[0:2]
    spike_t_dn = times[0:2]
    interpolate_from = 0.0
    interpolation_activation = 0
    intercept_point = 0

    for i, time in enumerate(times[1:]):
        slope = ((input_signal[i]-input_signal[i-1])/delta_time)
        if remainder_of_refractory >= 2*delta_time:
            remainder_of_refractory = remainder_of_refractory-delta_time
            interpolation_activation = 1
            continue

        if interpolation_activation == 1:
            interpolate_from = (interpolate_from+remainder_of_refractory)
            remainder_of_refractory = 0
            if interpolate_from >= 2*delta_time:
                interpolate_from = interpolate_from-delta_time
                continue
            interpolate_from = (
                interpolate_from+remainder_of_refractory) % delta_time
            voltage_below = (input_signal[i-1] + interpolate_from*slope)
            dc_voltage = voltage_below

        else:
            voltage_below = input_signal[i-1]
            interpolate_from = 0

        if dc_voltage + threshold_up <= input_signal[i]:
            intercept_point = time - delta_time + interpolate_from + \
                ((threshold_up+dc_voltage-voltage_below)/slope)
            spike_t_up = np.append(spike_t_up, intercept_point)
            interpolate_from = delta_time+intercept_point-time
            remainder_of_refractory = refractory_period_duration
            interpolation_activation = 1
            continue

        if dc_voltage - threshold_down >= input_signal[i]:
            intercept_point = time - delta_time + interpolate_from + \
                ((-threshold_down+dc_voltage-voltage_below)/slope)
            spike_t_dn = np.append(spike_t_dn, intercept_point)
            interpolate_from = delta_time+intercept_point-time
            remainder_of_refractory = refractory_period_duration
            interpolation_activation = 1
            continue

        interpolation_activation = 0

    index = [0, 1]
    spike_t_up = np.delete(spike_t_up, index)
    spike_t_dn = np.delete(spike_t_dn, index)

    return SpikeTrains(
        up=spike_t_up,
        down=spike_t_dn,
    )
