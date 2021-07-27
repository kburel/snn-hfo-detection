import numpy as np

# ========================================================================================
# Account for changes in a binary signal
# ========================================================================================


def detect_hfo(trial_duration, spike_monitor, spike_times, step_size, window_size):
    periods_of_hfo = np.array([[0, 0]])
    # ==============================================================================
    # Detect HFO
    # ==============================================================================
    assert step_size <= window_size
    # to get same number of time steps for all trials independently of spiking behaviour
    num_timesteps = int(np.ceil(trial_duration / step_size))

    # Prepare HFO signals
    hfo_identificaiton_signal = np.zeros(spike_times.size)

    mfr = np.zeros([num_timesteps])
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=trial_duration, step=step_size)):
        interval = [interval_start, interval_start + window_size]
        start_time, end_time = interval

        index = np.where(np.logical_and(
            spike_monitor >= start_time, spike_monitor <= end_time))[0]
        interval_duration = end_time - start_time
        mfr[interval_nr] = np.asarray(index.size / interval_duration)
        if index.size != 0:
            index_time_vector = np.where(np.logical_and(spike_times >= start_time,
                                                        spike_times <= end_time))[0]

            hfo_identificaiton_signal[index_time_vector] = 1

    signal_rise = []
    signal_fall = []

    binary_signal = hfo_identificaiton_signal

    for i in range(binary_signal.size-1):
        if i == 0 and binary_signal[0] == 1:
            signal_rise.append(i)
        if i > 0 and binary_signal[i] == 1 and binary_signal[i-1] == 0:
            signal_rise.append(i)
        elif binary_signal[i] == 1 and binary_signal[i+1] == 0:
            signal_fall.append(i)
        if i == binary_signal.size-2 and binary_signal[i] == 1:
            signal_fall.append(i)

    signal_rise = np.asarray(signal_rise)
    signal_fall = np.asarray(signal_fall)

    if signal_rise.size != 0:
        start_period_hfo = spike_times[signal_rise]
        stop_period_hfo = spike_times[signal_fall]
        periods_of_hfo = np.array([start_period_hfo, stop_period_hfo])
    else:
        periods_of_hfo = np.array([0, 0])

    hfo_detection = {}
    hfo_detection['total_hfo'] = signal_rise.size
    hfo_detection['time'] = spike_times
    hfo_detection['signal'] = hfo_identificaiton_signal
    hfo_detection['periods_hfo'] = periods_of_hfo

    return hfo_detection
