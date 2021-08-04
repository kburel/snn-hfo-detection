import matplotlib.pyplot as plt
from brian2.units import second, ms


class ChannelDebugError(Exception):
    def __init__(self, message, hfo_detection):
        super().__init__(message)
        self.hfo_detection = hfo_detection


def plot_internal_channel_debug(hfo_detection):
    raise ChannelDebugError(
        "plot_internal_channel_debug is just here for debugging purposes and should not be called",
        hfo_detection)


def plot_raster(hfo_detection):
    if hfo_detection.result.total_amount == 0:
        return
    plt.plot(hfo_detection.analytics.spike_times*second/ms,
           hfo_detection.analytics.neuron_ids, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')