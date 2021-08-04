from copy import deepcopy
from typing import List, NamedTuple, Optional
import numpy as np
from snn_hfo_ieeg.stages.all import run_all_hfo_detection_stages
from snn_hfo_ieeg.user_facing_data import HfoDetection, HfoDetectionWithAnalytics
from snn_hfo_ieeg.stages.loading.patient_data import load_patient_data, extract_channel_data
from snn_hfo_ieeg.stages.loading.folder_discovery import get_interval_paths
from snn_hfo_ieeg.stages.persistence.loading import load_hfo_detection
from snn_hfo_ieeg.plotting.persistence import persist_patient_plot, persist_channel_plot


class CustomOverrides(NamedTuple):
    duration: Optional[float]
    channels: Optional[List[int]]
    intervals: Optional[List[int]]


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


def _calculate_duration(signal_time):
    extra_simulation_time = 0.050
    return np.max(signal_time) + extra_simulation_time


def _generate_hfo_detection_cb(metadata, channel_data, duration, configuration, snn_cache):
    inner_configuration = deepcopy(configuration)
    inner_metada = deepcopy(metadata)
    if configuration.loading_path is not None:
        return lambda: load_hfo_detection(inner_configuration.loading_path, inner_metada)

    inner_channel_data = deepcopy(channel_data)
    inner_snn_cache = deepcopy(snn_cache)
    return lambda: run_all_hfo_detection_stages(
        metadata=inner_metada,
        channel_data=inner_channel_data,
        duration=duration,
        configuration=inner_configuration,
        snn_cache=inner_snn_cache)


def _generate_hfo_detector(metadata, channel_data, duration, configuration, snn_cache):
    hfo_detection_cb = _generate_hfo_detection_cb(
        metadata, channel_data, duration, configuration, snn_cache)
    return HfoDetector(hfo_detection_cb)


def run_hfo_detection_with_configuration(configuration, custom_overrides, hfo_cb):
    # Cache needs this lifetime
    snn_cache = None

    intervals = get_interval_paths(configuration.data_path)
    should_collect_patient_data = len(configuration.plots.patient) != 0
    patient_hfos = []
    for interval, interval_path in intervals.items():
        if custom_overrides.intervals is not None and interval not in custom_overrides.intervals:
            continue
        patient_data = load_patient_data(interval_path)
        duration = custom_overrides.duration if custom_overrides.duration is not None else _calculate_duration(
            patient_data.signal_time)

        for channel in range(len(patient_data.wideband_signals)):
            if custom_overrides.channels is not None and channel + 1 not in custom_overrides.channels:
                continue

            channel_data = extract_channel_data(patient_data, channel)
            metadata = Metadata(
                interval=interval,
                channel=channel + 1,
                channel_label=patient_data.channel_labels[channel],
                duration=duration
            )
            hfo_detector = _generate_hfo_detector(
                metadata, channel_data, duration, configuration, snn_cache)

            hfo_cb(metadata, hfo_detector)

            if hfo_detector.last_run is not None:
                for plotting_fn in configuration.plots.channel:
                    plotting_fn.function(hfo_detector.last_run)
                    persist_channel_plot(
                        plotting_fn.name, metadata, configuration)
                if should_collect_patient_data:
                    patient_hfos.append(
                        hfo_detector.last_run)

        for plotting_fn in configuration.plots.patient:
            plotting_fn.function(patient_hfos)
            persist_patient_plot(plotting_fn.name, configuration)
