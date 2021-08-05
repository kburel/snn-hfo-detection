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
    periods : Periods
        The start and end times in which HFOs were detected.
    filtered_spikes : FilteredSpikes
        The spike trains of the filtered bandwidths used for this detection
    spike_times : np.array
        Times when a neuron spiked.
    spike_times : np.array
        Times when a neuron spiked in seconds.
    neuron_ids : np.array
        The IDs of the neurons that fired at the time of spike_times. The indices match.
    '''
    detections: np.array
    periods: Periods
    filtered_spikes: FilteredSpikes
    spike_times: np.array
    neuron_ids: np.array


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


class PatientData(NamedTuple):
    '''
    Patient measurements
    '''
    wideband_signals: np.array
    signal_time: np.array
    channel_labels: np.array


class ChannelData(NamedTuple):
    '''
    Patient measurements for a specific channel
    '''
    wideband_signal: np.array
    signal_time: np.array


class Metadata(NamedTuple):
    interval: int
    channel: int
    channel_label: str
    duration: float


class HfoDetector():
    def __init__(self, hfo_detection_with_analytics_cb):
        self._hfo_detection_with_analytics_cb = hfo_detection_with_analytics_cb
        self.last_run = None

    def run(self) -> HfoDetection:
        return self.run_with_analytics().result

    def run_with_analytics(self) -> HfoDetectionWithAnalytics:
        if self.last_run is None:
            hfo_detection_with_analytics = self._hfo_detection_with_analytics_cb()
            self.last_run = hfo_detection_with_analytics
        return self.last_run


class HfoDetectionRun(NamedTuple):
    metadata: Metadata
    detector: HfoDetector
    input: ChannelData
