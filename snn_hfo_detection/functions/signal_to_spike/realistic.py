import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from snn_hfo_detection.functions.signal_to_spike.utility import SpikeTrains, get_sampling_frequency, SignalToSpikeParameters


def signal_to_spike(parameters: SignalToSpikeParameters) -> SpikeTrains:
    sampling_frequency = get_sampling_frequency(parameters.times)
    delta_time = 1/sampling_frequency
    times = parameters.times
    signal = parameters.signal
    if parameters.refractory_period < delta_time:
        interpolation_factor = 1
        while delta_time > parameters.refractory_period:
            interpolation_factor += 1
            delta_time = 1/(sampling_frequency*interpolation_factor)
        interpolation = interp1d(times, signal)
        times = np.concatenate(
            (np.arange(0, times[-1], delta_time), [times[-1]]))
        signal = interpolation(times)
        sampling_frequency = 1/times[1]
    interpolated_parameters = SignalToSpikeParameters(
        signal=signal,
        threshold_up=parameters.threshold_up,
        threshold_down=parameters.threshold_down,
        times=times,
        refractory_period=parameters.refractory_period)
    return _signal_to_spike(interpolated_parameters)


@ njit(fastmath=True, parallel=True)
def _signal_to_spike(parameters: SignalToSpikeParameters) -> SpikeTrains:
    delta_time = parameters.times[1] - parameters.times[0]
    dc_voltage = parameters.signal[0]
    remainder_of_refractory = 0
    spike_t_up = parameters.times[0:2]
    spike_t_dn = parameters.times[0:2]
    interpolate_from = 0.0
    interpolation_activation = 0
    intercept_point = 0

    for i, time in enumerate(parameters.times[1:]):
        slope = (
            (parameters.signal[i]-parameters.signal[i-1])/delta_time)
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
            voltage_below = (
                parameters.signal[i-1] + interpolate_from*slope)
            dc_voltage = voltage_below

        else:
            voltage_below = parameters.signal[i-1]
            interpolate_from = 0

        if dc_voltage + parameters.threshold_up <= parameters.signal[i]:
            intercept_point = time - delta_time + interpolate_from + \
                ((parameters.threshold_up+dc_voltage-voltage_below)/slope)
            spike_t_up = np.append(spike_t_up, intercept_point)
            interpolate_from = delta_time+intercept_point-time
            remainder_of_refractory = parameters.refractory_period
            interpolation_activation = 1
            continue

        if dc_voltage - parameters.threshold_down >= parameters.signal[i]:
            intercept_point = time - delta_time + interpolate_from + \
                ((-parameters.threshold_down+dc_voltage-voltage_below)/slope)
            spike_t_dn = np.append(spike_t_dn, intercept_point)
            interpolate_from = delta_time+intercept_point-time
            remainder_of_refractory = parameters.refractory_period
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
