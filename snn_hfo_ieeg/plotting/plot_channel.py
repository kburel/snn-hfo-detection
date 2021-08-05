import matplotlib.pyplot as plt
from brian2.units import second, ms
from snn_hfo_ieeg.user_facing_data import HfoDetectionRun


class ChannelDebugError(Exception):
    def __init__(self, message, hfo_run: HfoDetectionRun):
        super().__init__(message)
        self.hfo_run = hfo_run


def plot_internal_channel_debug(hfo_run: HfoDetectionRun):
    raise ChannelDebugError(
        "plot_internal_channel_debug is just here for debugging purposes and should not be called",
        hfo_run)


def plot_raster(hfo_run: HfoDetectionRun):
    hfo_detection = hfo_run.detector.last_run
    if hfo_detection.result.total_amount == 0:
        return
    plt.plot(hfo_detection.analytics.spike_times*second/ms,
             hfo_run.analytics.neuron_ids, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
