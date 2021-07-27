import numpy as np
from typing import NamedTuple

# ========================================================================================
# Account for changes in a binary signal
# ========================================================================================


class HfoPeriod(NamedTuple):
    start: float
    stop: float


def get_binary_hfos(duration, spike_times, signal_times, step_size, window_size):
    binary_hfo_signal = np.zeros(len(signal_times)).astype(int)

    for start_time in np.arange(start=0, stop=duration, step=step_size):
        end_time = start_time + window_size

        are_spike_times_in_window = np.any(
            (spike_times >= start_time) & (spike_times <= end_time))
        if are_spike_times_in_window:
            index_time_vector = np.where(
                (signal_times >= start_time) & (signal_times <= end_time))

            binary_hfo_signal[index_time_vector] = 1
    return binary_hfo_signal


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

    binary_hfo_signal = get_binary_hfos(
        duration, spike_times, signal_times, step_size, window_size)
    periods = find_periods(binary_hfo_signal, signal_times)
    flat_periods = _flatten_periods(periods)

    hfo_detection = {}
    hfo_detection['total_hfo'] = len(periods)
    hfo_detection['time'] = signal_times
    hfo_detection['signal'] = binary_hfo_signal
    hfo_detection['periods_hfo'] = flat_periods
    return hfo_detection
