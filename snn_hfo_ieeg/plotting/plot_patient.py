from typing import TypedDict, NamedTuple
import matplotlib.pyplot as plt
from snn_hfo_ieeg.user_facing_data import HfoDetectionWithAnalytics, Metadata


class ChannelData(NamedTuple):
    metadata: Metadata
    hfo_detection: HfoDetectionWithAnalytics


class Intervals(TypedDict):
    index: int
    channel_data: ChannelData


class PatientDebugError(Exception):
    def __init__(self, message, intervals: Intervals):
        super().__init__(message)
        self.intervals = intervals


def plot_internal_patient_debug(intervals: Intervals):
    raise PatientDebugError(
        "plot_internal_patient_debug is just here for debugging purposes and should not be called",
        intervals)
