import matplotlib.pyplot as p
from brian2.units import second, ms


def plot_raster(hfo_detection):
    if hfo_detection.result.total_amount == 0:
        return
    p.plot(hfo_detection.analytics.spike_times*second/ms,
           hfo_detection.analytics.neuron_ids, '.k')
    p.xlabel('Time (ms)')
    p.ylabel('Neuron index')
    p.show()
