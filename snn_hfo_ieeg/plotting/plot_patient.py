from typing import List, TypedDict, NamedTuple
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from snn_hfo_ieeg.user_facing_data import HfoDetectionWithAnalytics, Metadata
color_bars = ['#2f70b6', '#c85a2d', '#6f63b6', '#e26a84']


class ChannelData(NamedTuple):
    metadata: Metadata
    hfo_detection: HfoDetectionWithAnalytics


class Intervals(TypedDict):
    index: int
    channel_data: List[ChannelData]


class PatientDebugError(Exception):
    def __init__(self, message, intervals: Intervals):
        super().__init__(message)
        self.intervals = intervals


def plot_internal_patient_debug(intervals: Intervals):
    raise PatientDebugError(
        "plot_internal_patient_debug is just here for debugging purposes and should not be called",
        intervals)


def _append_or_create(dict, key, value):
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)


def plot_mean_hfo_rate(intervals: Intervals):
    label_to_hfo_rates = {}
    print(intervals)
    for channels in intervals.values():
        for channel in channels:
            _append_or_create(
                dict=label_to_hfo_rates,
                key=channel.metadata.channel_label,
                value=channel.hfo_detection.result.frequency * 60)
    labels = label_to_hfo_rates.keys()
    hfo_rates = label_to_hfo_rates.values()
    mean_hfo_rates = [mean(_) for _ in hfo_rates]
    standard_deviations = [np.std(_) for _ in hfo_rates]
    plt.ylabel('Average HFO rate per minute')
    plt.xlabel('Channel')
    plt.bar(
        x=labels,
        height=mean_hfo_rates,
        yerr=standard_deviations,
        capsize=2)
