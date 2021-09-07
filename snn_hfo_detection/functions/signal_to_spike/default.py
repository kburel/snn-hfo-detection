import numpy as np
from scipy.interpolate import interp1d
from snn_hfo_detection.functions.signal_to_spike.utility import SpikeTrains, SignalToSpikeParameters


def signal_to_spike(parameters: SignalToSpikeParameters):
    '''
    This functions retuns two spike trains, when the signal crosses the specified threshold in
    a rising direction (UP spikes) and when it crosses the specified threshold in a falling
    direction (DOWN spikes)
    :times (array): time vector
    :amplitude (array): amplitude of the signal
    :interpolation_factor (int): upsampling factor, new sampling frequency
    :thr_up (float): threshold crossing in a rising direction
    :thr_dn (float): threshold crossing in a falling direction
    :refractory_period (float): period in which no spike will be generated [same units as time vector]
    '''
    actual_dc = 0
    spike_up = []
    spike_dn = []

    intepolated_time = interp1d(parameters.times, parameters.signal)
    rangeint = np.round(
        (np.max(parameters.times) - np.min(parameters.times))*parameters.interpolation_factor)
    xnew = np.linspace(np.min(parameters.times), np.max(
        parameters.times), num=int(rangeint), endpoint=True)
    data = np.reshape([xnew, intepolated_time(xnew)], (2, len(xnew))).T

    i = 0
    while i < (len(data)):
        if((actual_dc + parameters.threshold_up) < data[i, 1]):
            spike_up.append(data[i, 0])  # spike up
            actual_dc = data[i, 1]        # update current dc value
            i += int(parameters.refractory_period *
                     parameters.interpolation_factor)
        elif((actual_dc - parameters.threshold_down) > data[i, 1]):
            spike_dn.append(data[i, 0])  # spike dn
            actual_dc = data[i, 1]        # update curre
            i += int(parameters.refractory_period *
                     parameters.interpolation_factor)
        else:
            i += 1

    return SpikeTrains(up=spike_up,
                       down=spike_dn)
