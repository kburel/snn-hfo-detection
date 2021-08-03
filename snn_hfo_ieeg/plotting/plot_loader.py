import re
from typing import NamedTuple, Callable, List
from inspect import getmembers, isfunction
from snn_hfo_ieeg.plotting import plot_channel
from snn_hfo_ieeg.plotting import plot_patient

PLOTTING_REGEX = re.compile(r'^plot_([\w_]+)$')


class PlottingFunction(NamedTuple):
    name: str
    function: Callable


class PlottingFunctions(NamedTuple):
    channel: List[PlottingFunction]
    patient: List[PlottingFunction]


def _get_plotting_functions(module):
    all_names_and_functions = getmembers(module, isfunction)
    potential_matches = [PLOTTING_REGEX.match(
        name) for name, _fn in all_names_and_functions]
    functions = [fn for _name, fn in all_names_and_functions]
    actual_matches = [(potential_match.group(1), function)
                      for potential_match, function in zip(potential_matches, functions)
                      if potential_match is not None]
    return [PlottingFunction(name, function) for name, function in actual_matches]


def _get_available_plotting_functions():
    return PlottingFunctions(
        channel=_get_plotting_functions(plot_channel),
        patient=_get_plotting_functions(plot_patient)
    )


def _find_name(name, plotting_functions):
    return next(
        (fn for fn in plotting_functions if fn.name == name), None)


def find_plotting_functions(plot_names):
    unique_plot_names = list(set(plot_names))

    channel_fns = []
    patient_fns = []
    plotting_functions = _get_available_plotting_functions()
    for plot_name in unique_plot_names:
        channel_fn = _find_name(plot_name, plotting_functions.channel)
        patient_fn = _find_name(plot_name, plotting_functions.patient)

        if channel_fn is None and patient_fn is None:
            raise ValueError(
                f'run.py: error: the desired plot "{plot_name}" was not found.')
        if channel_fn is not None:
            channel_fns.append(channel_fn)
        if patient_fn is not None:
            patient_fns.append(patient_fn)

    return PlottingFunctions(channel=channel_fns, patient=patient_fns)
