from typing import NamedTuple
import numpy as np
from numpy.linalg import det

# ========================================================================================
# Account for changes in a binary signal
# ========================================================================================


class Window(NamedTuple):
    start: float
    stop: float


class Periods(NamedTuple):
    '''
    Periods in which HFOs were detected. Used for plotting.

    Parameters
    -----
    start : np.array
        List of times where an HFO started.
    start : np.array
        List of times where an HFO stopped.
    '''
    start: np.array
    stop: np.array


class PlottingData(NamedTuple):
    '''
    Convenience data that can be used for plotting.

    Parameters
    -----
    detections : np.array
        Boolean list of HFO detection. The indices correspond to analyzed_times.
    analyzed_times : np.array
        List of all analyzed timestamps.
    periods : Periods
        The start and end times in which HFOs were detected.
    '''
    detections: np.array
    analyzed_times: np.array
    periods: Periods


class HfoDetection(NamedTuple):
    '''
    The result of the SNNs HFO detection

    Parameters
    -----
    frequency : float
        The measured frequency of HFOs. This is the most important value.
    total_amount : int
        Total amount of detected HFOs over the entire dataset.
    plotting_data : PlottingData
        Convenience data that a user can use for their plotting.
    '''
    frequency: float
    total_amount: int
    plotting_data: PlottingData


def _did_snn_find_hfo(spike_times, window):
    return np.any(
        (spike_times >= window.start) & (spike_times <= window.end))


def _get_time_indices_in_window(signal_times, window):
    return np.where(
        (signal_times >= window.start) & (signal_times <= window.end))


def get_binary_hfos(duration, spike_times, signal_times, step_size, window_size):
    binary_hfo_signal = np.zeros(len(signal_times)).astype(int)

    for start_time in np.arange(start=0, stop=duration, step=step_size):
        window = Window(start=start_time, end=start_time + window_size)

        if _did_snn_find_hfo(spike_times, window):
            hfo_indices = _get_time_indices_in_window(
                signal_times, window)
            binary_hfo_signal[hfo_indices] = 1
    return binary_hfo_signal


def find_periods(signals, times):
    periods = []
    for signal, time in zip(signals, times):
        is_last_period_finished = len(
            periods) == 0 or periods[-1].stop is not None

        if signal == 0 and not is_last_period_finished:
            periods[-1] = Window(start=periods[-1].start, stop=time)
        if signal == 1 and is_last_period_finished:
            periods.append(Window(start=time, stop=None))
    return periods


def _flatten_periods(periods):
    start = [period.start for period in periods]
    stop = [period.stop for period in periods if period.stop is not None]
    return Periods(start, stop)


def detect_hfo(duration, spike_times, signal_times, step_size, window_size):
    assert step_size <= window_size

    binary_hfo_signal = get_binary_hfos(
        duration, spike_times, signal_times, step_size, window_size)
    periods = find_periods(binary_hfo_signal, signal_times)
    flat_periods = _flatten_periods(periods)

    return HfoDetection(
        total_amount=len(periods),
        frequency=len(periods)/duration,
        plotting_data=PlottingData(
            detections=binary_hfo_signal,
            analyzed_times=signal_times,
            periods=flat_periods
        )
    )
