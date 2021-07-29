from enum import Enum, auto
from snn_hfo_ieeg.plotting.channel.plot_dummy import plot_dummy


class PlotKind(Enum):
    TODO = auto()


def _get_plotting_fn_for_total(_plot_kind):
    return None


def _get_plotting_fn_for_channel(plot_kind):
    if plot_kind is PlotKind.TODO:
        return plot_dummy
    return None


def get_plotting_fns(plot_kinds, get_plotting_fn_cb):
    plot_fns = [get_plotting_fn_cb(plot_kind) for plot_kind in plot_kinds]
    return [plot_fn for plot_fn in plot_fns if plot_fns is not None]


def get_plotting_fns_for_channel(plot_kinds):
    return get_plotting_fns(plot_kinds, _get_plotting_fn_for_channel)


def get_plotting_fn_for_total(plot_kinds):
    return get_plotting_fns(plot_kinds, _get_plotting_fn_for_total)
