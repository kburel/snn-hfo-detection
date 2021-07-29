from matplotlib.pyplot import plot, xlabel, ylabel
from brian2.units import ms


def plot_raster(hfo_detection):
    plot(hfo_detection.analytics.spike_times/ms,
         hfo_detection.analytics.neuron_ids, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
