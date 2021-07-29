from snn_hfo_ieeg.stages.plotting.plot_factory import get_plotting_fns_for_channel, get_plotting_fns_for_total


def plot_for_channel(hfo_detection_with_analytics, plotting_kinds):
    for plotting_fn in get_plotting_fns_for_channel(plotting_kinds):
        plotting_fn(hfo_detection_with_analytics)


def plot_for_total(hfo_detections_with_analytics, plotting_kinds):
    for plotting_fn in get_plotting_fns_for_total(plotting_kinds):
        plotting_fn(hfo_detections_with_analytics)
