from typing import List, NamedTuple
import numpy as np
from snn_hfo_ieeg.functions.hfo_detection import HfoDetection
from snn_hfo_ieeg.stages.all import run_all_hfo_detection_stages
from snn_hfo_ieeg.stages.loading.patient_data import load_patient_data, extract_channel_data
from snn_hfo_ieeg.stages.loading.folder_discovery import get_patient_interval_paths


class CustomOverrides(NamedTuple):
    duration: float
    channels: List[int]
    patients: List[int]
    intervals: List[int]


class HfoDetectionRun(NamedTuple):
    patient: int
    interval: int
    channel: int
    duration: float
    hfo_detection: HfoDetection


def _calculate_duration(signal_time):
    extra_simulation_time = 0.050
    return np.max(signal_time) + extra_simulation_time


def run_hfo_detection_with_configuration(configuration, custom_overrides, hfo_cb):
    patient_intervals_paths = get_patient_interval_paths(
        configuration.data_path)
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
                hfo_detection = run_all_hfo_detection_stages(
                    channel_data=channel_data,
                    duration=duration,
                    configuration=configuration)

                hfo_detection_run = HfoDetectionRun(
                    patient=patient,
                    interval=interval,
                    channel=channel + 1,
                    hfo_detection=hfo_detection,
                    duration=duration
                )

                hfo_cb(hfo_detection_run)
