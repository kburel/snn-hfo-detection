import numpy as np
from typing import NamedTuple

# ========================================================================================
# Account for changes in a binary signal
# ========================================================================================


class HfoPeriod(NamedTuple):
    start: float
    stop: float


def find_periods(signals, times):
    periods = []
    for signal, time in zip(signals, times):
        is_last_period_finished = len(
            periods) == 0 or periods[-1].stop is not None

        if signal == 0 and not is_last_period_finished:
            periods[-1] = HfoPeriod(start=periods[-1].start, stop=time)
        if signal == 1 and is_last_period_finished:
            periods.append(HfoPeriod(start=time, stop=None))
    return periods


def _flatten_periods(periods):
    start = [period.start for period in periods]
    stop = [period.stop for period in periods if period.stop is not None]
    return [start, stop]


def detect_hfo(duration, spike_times, signal_times, step_size, window_size):
    # ==============================================================================
    # Detect HFO
    # ==============================================================================
    assert step_size <= window_size

    # Prepare HFO signals
    binary_hfo_signal = np.zeros(len(signal_times)).astype(int)

    for interval_start in np.arange(start=0, stop=duration, step=step_size):
        interval = [interval_start, interval_start + window_size]
        start_time, end_time = interval

        index = np.where(np.logical_and(
            spike_times >= start_time, spike_times <= end_time))[0]
        if index.size != 0:
            index_time_vector = np.where(np.logical_and(signal_times >= start_time,
                                                        signal_times <= end_time))[0]

            binary_hfo_signal[index_time_vector] = 1

    periods = find_periods(binary_hfo_signal, signal_times)
    flat_periods = _flatten_periods(periods)

    hfo_detection = {}
    hfo_detection['total_hfo'] = len(periods)
    hfo_detection['time'] = signal_times
    hfo_detection['signal'] = binary_hfo_signal
    hfo_detection['periods_hfo'] = flat_periods
    return hfo_detection
