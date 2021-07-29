from typing import NamedTuple, Optional
import numpy as np


class SpikeTrains(NamedTuple):
    '''
    Up and down spike trains received by filtering a signal
    '''
    up: np.array
    down: np.array


class FilteredSpikes(NamedTuple):
    '''
    Spikes in the filtered bandwidths. If some of these are None, it means
    that they are not suited for analysis in the specified MeasurementMode

    Parameters
    -------
    ripple : Optional[SpikeTrains]
        Spikes in the ripple bandwidth (80-250 Hz).
    fast_ripple: Optional[SpikeTrains]
        Spikes in the fast ripple bandwidth (250-500 Hz).
    '''
    ripple: Optional[SpikeTrains]
    fast_ripple: Optional[SpikeTrains]


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
        Boolean list of HFO detection. The indices correspond to analyzed_times.
    analyzed_times : np.array
        List of all analyzed timestamps.
    periods : Periods
        The start and end times in which HFOs were detected.
    filtered_spikes : FilteredSpikes
        The spike trains of the filtered bandwidths used for this detection

    '''
    detections: np.array
    analyzed_times: np.array
    periods: Periods
    filtered_spikes: FilteredSpikes


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
