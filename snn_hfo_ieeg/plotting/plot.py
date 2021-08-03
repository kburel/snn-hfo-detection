from enum import Enum, auto
from os import path, makedirs
import matplotlib.pyplot as plt


class PlotMode(Enum):
    SAVE = auto()
    SHOW = auto()
    BOTH = auto()


def _save_plot(plot_name, parent_directory):
    makedirs(parent_directory, exist_ok=True)
    filename = f'{plot_name}.png'
    full_path = path.join(parent_directory, filename)
    plt.savefig(full_path)


def _should_show_plot(configuration):
    return configuration.plot_mode is PlotMode.SAVE or configuration.plot_mode is PlotMode.BOTH


def _should_save_plot(configuration):
    return configuration.plot_mode is PlotMode.SAVE or configuration.plot_mode is PlotMode.BOTH


def persist_channel_plot(plot_name, metadata, configuration):
    if _should_show_plot(configuration):
        plt.show()
    if _should_save_plot(configuration):
        parent_dir = path.join(
            configuration.plot_path, metadata.patient, metadata.interval, metadata.channel)
        _save_plot(plot_name, parent_dir)


def persist_patient_plot(plot_name, patient, configuration):
    if _should_show_plot(configuration):
        plt.show()
    if _should_save_plot(configuration):
        parent_dir = path.join(configuration.plot_path, patient)
        _save_plot(plot_name, parent_dir)
