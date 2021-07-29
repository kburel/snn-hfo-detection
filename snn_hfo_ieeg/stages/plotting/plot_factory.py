from enum import Enum, auto
from typing import NamedTuple, List
from snn_hfo_ieeg.stages.plotting.channel.plot_channel_dummy import plot_internal_channel_debug
from snn_hfo_ieeg.stages.plotting.total.plot_total_dummy import plot_internal_total_debug


class ChannelPlotKind(Enum):
    INTERNAL_CHANNEL_DEBUG = auto()


class TotalPlotKind(Enum):
    INTERNAL_TOTAL_DEBUG = auto()


class Plots(NamedTuple):
    channel: List[ChannelPlotKind]
    total: List[ChannelPlotKind]


def _get_plotting_fn_for_channel(plot_kind):
    if plot_kind is ChannelPlotKind.INTERNAL_CHANNEL_DEBUG:
        return plot_internal_channel_debug
    raise ValueError(
        f'plot_kind must be valid ChannelPlotKind. Expected one of {ChannelPlotKind}, but got {plot_kind}')


def _get_plotting_fn_for_total(plot_kind):
    if plot_kind is TotalPlotKind.INTERNAL_TOTAL_DEBUG:
        return plot_internal_total_debug
    raise ValueError(
        f'plot_kind must be valid TotalPlotKind. Expected one of {TotalPlotKind}, but got {plot_kind}')


def get_plotting_fns(plot_kinds, get_plotting_fn_cb):
    plot_fns = [get_plotting_fn_cb(plot_kind) for plot_kind in plot_kinds]
    return [plot_fn for plot_fn in plot_fns if plot_fns is not None]


def get_plotting_fns_for_channel(plot_kinds):
    return get_plotting_fns(plot_kinds, _get_plotting_fn_for_channel)


def get_plotting_fns_for_total(plot_kinds):
    return get_plotting_fns(plot_kinds, _get_plotting_fn_for_total)
