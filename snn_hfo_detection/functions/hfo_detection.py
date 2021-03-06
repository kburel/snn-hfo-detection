from typing import NamedTuple
import numpy as np

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


class Analytics(NamedTuple):
    '''
    Convenience data that can be used for plotting.

    Parameters
    -----
    detections : np.array
        Boolean list of HFO detection. The indices correspond to the input's analyzed times.
    periods : Periods
        The start and end times in which HFOs were detected.
    '''
    detections: np.array
    periods: Periods


class HfoDetection(NamedTuple):
    '''
    The result of the SNNs HFO detection

    Parameters
    -----
    frequency : float
        The measured frequency of HFOs per second. This is the most important value.
    total_amount : int
        Total amount of detected HFOs over the entire dataset.
    '''
    frequency: float
    total_amount: int


class HfoDetectionWithAnalytics(NamedTuple):
    '''
    The result of the SNNs HFO detection

    Parameters
    -----
    result : HfoDetection
        The HFO detection result
    analytics : Analytics
        Convenience data that a user can use for their plotting.
    '''
    result: HfoDetection
    analytics: Analytics


def _did_snn_find_hfo(spike_times, window):
    return np.any(
        (spike_times >= window.start) & (spike_times <= window.stop))


def _get_time_indices_in_window(signal_times, window):
    return np.where(
        (signal_times >= window.start) & (signal_times <= window.stop))


def get_binary_hfos(duration, spike_times, signal_times, step_size, window_size):
    binary_hfo_signal = np.zeros(len(signal_times)).astype(bool)

    for start_time in np.arange(start=0, stop=duration, step=step_size):
        window = Window(start=start_time,
                        stop=(start_time + window_size))

        if _did_snn_find_hfo(spike_times, window):
            hfo_indices = _get_time_indices_in_window(
                signal_times, window)
            binary_hfo_signal[hfo_indices] = True
    return binary_hfo_signal


def _find_periods(signals, times):
    if len(signals) == 0:
        raise ValueError('signals is not allowed to be empty, but was')
    if len(times) == 0:
        raise ValueError('times is not allowed to be empty, but was')
    if len(signals) != len(times):
        raise ValueError(
            f'signals and times need to have corresponding indices, but signals has length {len(signals)} while times has length {len(times)}')

    periods = []
    for signal, time in zip(signals, times):
        is_last_period_finished = len(
            periods) == 0 or periods[-1].stop is not None

        if signal == 0 and not is_last_period_finished:
            periods[-1] = Window(start=periods[-1].start, stop=time)
        if signal == 1 and is_last_period_finished:
            periods.append(Window(start=time, stop=None))
    if len(periods) != 0 and periods[-1].stop is None:
        periods[-1] = Window(periods[-1].start, times[-1])
    return periods


def _flatten_periods(periods):
    start = [period.start for period in periods]
    stop = [period.stop for period in periods if period.stop is not None]
    return Periods(start, stop)


def detect_hfo(duration, spike_times, signal_times, step_size, window_size):
    if step_size > window_size:
        raise ValueError(
            f'step_size needs to be at most windows_size, but got: step_size={step_size}, window_size={step_size}')
    if duration <= 0:
        raise ValueError(
            f'Tried to detect an HFO for a dataset with a duration that under or equal to zero. Got duration: {duration}')

    binary_hfo_signal = get_binary_hfos(
        duration, spike_times, signal_times, step_size, window_size)
    periods = _find_periods(binary_hfo_signal, signal_times)
    flat_periods = _flatten_periods(periods)

    return HfoDetectionWithAnalytics(
        result=HfoDetection(
            total_amount=len(periods),
            frequency=len(periods)/duration,
        ),
        analytics=Analytics(
            detections=binary_hfo_signal,
            periods=flat_periods
        )
    )
