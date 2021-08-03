import pytest
from snn_hfo_ieeg.plotting.plot_loader import find_plotting_functions
from snn_hfo_ieeg.plotting.plot_channel import ChannelDebugError

VALID_PLOT_NAME = 'internal_channel_debug'


def test_fails_on_missing_plot():
    with pytest.raises(ValueError):
        find_plotting_functions(['yarrr'])


def test_fails_on_empty_plot_name():
    with pytest.raises(ValueError):
        find_plotting_functions([''])


def test_returns_no_plots_for_no_names():
    plots = find_plotting_functions([])
    assert len(plots.channel) == 0
    assert len(plots.patient) == 0


def test_returns_correct_plot_name():
    plots = find_plotting_functions([VALID_PLOT_NAME])
    assert len(plots.channel) == 1
    assert len(plots.patient) == 0
    assert plots.channel[0].name == VALID_PLOT_NAME


def test_return_scorrect_plot_fn():
    plots = find_plotting_functions([VALID_PLOT_NAME])
    assert len(plots.channel) == 1
    assert len(plots.patient) == 0
    print(plots.channel[0].function)
    with pytest.raises(ChannelDebugError):
        plots.channel[0].function(None)


def test_returns_plots_with_multiple_names():
    plots = find_plotting_functions(
        [VALID_PLOT_NAME, 'internal_patient_debug'])
    assert len(plots.channel) == 1
    assert len(plots.patient) == 1


def test_returns_single_plot_with_duplicate_names():
    plots = find_plotting_functions([VALID_PLOT_NAME, VALID_PLOT_NAME])
    assert len(plots.channel) == 1
    assert len(plots.patient) == 0
