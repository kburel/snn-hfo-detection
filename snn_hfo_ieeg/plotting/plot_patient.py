from typing import List, TypedDict
from snn_hfo_ieeg.user_facing_data import HfoDetectionRun
from snn_hfo_ieeg.plotting.plot_mean_hfo_rate import plot_mean_hfo_rate as inner_plot_mean_hfo_rate


class Intervals(TypedDict):
    index: int
    channel_data: List[HfoDetectionRun]


class PatientDebugError(Exception):
    def __init__(self, message, intervals: Intervals):
        super().__init__(message)
        self.intervals = intervals


def plot_internal_patient_debug(intervals: Intervals):
    raise PatientDebugError(
        "plot_internal_patient_debug is just here for debugging purposes and should not be called",
        intervals)


def plot_mean_hfo_rate(intervals: Intervals):
    inner_plot_mean_hfo_rate(intervals)
