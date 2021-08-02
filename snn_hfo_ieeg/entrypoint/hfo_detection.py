from copy import deepcopy
from typing import List, NamedTuple, Optional
import numpy as np
from snn_hfo_ieeg.stages.all import run_all_hfo_detection_stages
from snn_hfo_ieeg.user_facing_data import HfoDetection, HfoDetectionWithAnalytics
from snn_hfo_ieeg.stages.loading.patient_data import load_patient_data, extract_channel_data
from snn_hfo_ieeg.stages.loading.folder_discovery import get_patient_interval_paths


class CustomOverrides(NamedTuple):
    duration: Optional[float]
    channels: Optional[List[int]]
    patients: Optional[List[int]]
    intervals: Optional[List[int]]


class Metadata(NamedTuple):
    patient: int
    interval: int
    channel: int
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
    inner_channel_data = deepcopy(channel_data)
    inner_configuration = deepcopy(configuration)
    inner_metada = deepcopy(metadata)
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

    patient_intervals_paths = get_patient_interval_paths(
        configuration.data_path)
    should_collect_patient_data = len(configuration.plots.patient) != 0
    patient_hfos = []
    for patient, intervals in patient_intervals_paths.items():
        if custom_overrides.patients is not None and patient not in custom_overrides.patients:
            continue
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
                    patient=patient,
                    interval=interval,
                    channel=channel + 1,
                    duration=duration
                )
                hfo_detector = _generate_hfo_detector(
                    metadata, channel_data, duration, configuration, snn_cache)

                hfo_cb(metadata, hfo_detector)
                if should_collect_patient_data and hfo_detector.last_run is not None:
                    patient_hfos.append(
                        hfo_detector.last_run)

        for _, plotting_function in configuration.plots.patient:
            plotting_function(patient_hfos)
