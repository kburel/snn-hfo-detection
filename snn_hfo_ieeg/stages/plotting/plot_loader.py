import re
from typing import NamedTuple, Callable, List
from inspect import getmembers, isfunction
from snn_hfo_ieeg.stages.plotting import plot_channel
from snn_hfo_ieeg.stages.plotting import plot_total

PLOTTING_REGEX = re.compile(r'^plot_([\w_]+)$')


class PlottingFunction(NamedTuple):
    name: str
    function: Callable


class PlottingFunctions(NamedTuple):
    channel: List[PlottingFunction]
    total: List[PlottingFunction]


def _get_plotting_functions(module):
    all_functions = getmembers(module, isfunction)
    potential_matches = [PLOTTING_REGEX.match(
        name) for name, _fn in all_functions]
    actual_matches = [(potential_match.group(1), function)
                      for potential_match, function in zip(potential_matches, all_functions)
                      if potential_match is not None]
    return [PlottingFunction(name, function) for name, function in actual_matches]


def _get_available_plotting_functions():
    return PlottingFunctions(
        channel=_get_plotting_functions(plot_channel),
        total=_get_plotting_functions(plot_total)
    )


def _find_name(name, plotting_functions):
    print(plotting_functions)
    return next(
        (fn for fn in plotting_functions if fn.name == name), None)


def find_plotting_functions(plot_names):
    channel_fns = []
    total_fns = []
    plotting_functions = _get_available_plotting_functions()
    for plot_name in plot_names:
        print(plot_name)
        print(plotting_functions.channel)
        channel_fn = _find_name(plot_name, plotting_functions.channel)
        total_fn = _find_name(plot_name, plotting_functions.total)
        if channel_fn is None and total_fn is None:
            raise ValueError(
                f'run.py: error: the desired plot "{plot_name}" was not found.')
        if channel_fn is not None:
            channel_fns.append(channel_fn)
        if total_fn is not None:
            total_fns.append(total_fn)

    return PlottingFunctions(channel=channel_fns, total=total_fns)
