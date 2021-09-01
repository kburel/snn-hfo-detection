from typing import NamedTuple, Optional, Callable, List
from enum import Enum, auto
import numpy as np


class SpikeTrains(NamedTuple):
    '''
    Up and down spike trains received by filtering a signal
    '''
    up: np.array
    down: np.array


class Bandwidth(NamedTuple):
    signal: np.array
    spike_trains: SpikeTrains


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
    very_fast_ripple: Optional[SpikeTrains]
        Spikes in the very fast ripple bandwidth (500-900 Hz).
    '''
    ripple: Optional[Bandwidth]
    fast_ripple: Optional[Bandwidth]
    very_fast_ripple: Optional[Bandwidth]


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
    last_run: Optional[HfoDetectionWithAnalytics]

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


class MeasurementMode(Enum):
    IEEG = auto()
    ECOG = auto()
    SCALP = auto()


class PlottingFunction(NamedTuple):
    name: str
    function: Callable


class PlottingFunctions(NamedTuple):
    channel: List[PlottingFunction]
    patient: List[PlottingFunction]


class PlotMode(Enum):
    SAVE = auto()
    SHOW = auto()
    BOTH = auto()


class Configuration(NamedTuple):
    data_path: str
    measurement_mode: MeasurementMode
    hidden_neuron_count: int
    calibration_time: float
    plots: PlottingFunctions
    saving_path: Optional[str]
    disable_saving: bool
    loading_path: Optional[str]
    plot_path: str
    plot_mode: PlotMode


class HfoDetectionRun(NamedTuple):
    metadata: Metadata
    detector: HfoDetector
    input: ChannelData
    configuration: Configuration
