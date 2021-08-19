from os import path, makedirs
from typing import List, TypedDict
import matplotlib.pyplot as plt
from snn_hfo_ieeg.user_facing_data import HfoDetectionRun
from snn_hfo_ieeg.user_facing_data import HfoDetectionWithAnalytics, PlotMode


class Intervals(TypedDict):
    index: int
    channel_data: List[HfoDetectionRun]


def _save_plot(plot_name, parent_directory):
    makedirs(parent_directory, exist_ok=True)
    filename = f'{plot_name}.png'
    full_path = path.join(parent_directory, filename)
    plt.savefig(full_path)


def should_show_plot(configuration):
    return configuration.plot_mode is PlotMode.SHOW or configuration.plot_mode is PlotMode.BOTH


def should_save_plot(configuration):
    return configuration.plot_mode is PlotMode.SAVE or configuration.plot_mode is PlotMode.BOTH


def save_or_show_channel_plot(plot_name, hfo_detection_run: HfoDetectionWithAnalytics):
    if should_show_plot(hfo_detection_run.configuration):
        plt.show()
    if should_save_plot(hfo_detection_run.configuration):
        parent_dir = path.join(
            hfo_detection_run.configuration.plot_path, f'I{hfo_detection_run.metadata.interval}', f'C{hfo_detection_run.metadata.channel}')
        _save_plot(plot_name, parent_dir)


def save_or_show_patient_plot(plot_name, intervals: Intervals):
    configuration = next(iter(intervals.values())).channel_data.configuration
    if should_show_plot(configuration):
        plt.show()
    if should_save_plot(configuration):
        _save_plot(plot_name, configuration.plot_path)
