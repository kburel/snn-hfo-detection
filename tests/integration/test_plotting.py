import pytest
from snn_hfo_ieeg.stages.plotting.total.plot_total_dummy import TotalDebugError
from snn_hfo_ieeg.stages.plotting.channel.plot_channel_dummy import ChannelDebugError
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode
from snn_hfo_ieeg.entrypoint.hfo_detection import run_hfo_detection_with_configuration
from snn_hfo_ieeg.stages.plotting.plot_factory import ChannelPlotKind, Plots, TotalPlotKind
from tests.integration.utility import get_hfo_directory, EMPTY_CUSTOM_OVERRIDES


def _hfo_runner_cb(_metadata, hfo_detector):
    hfo_detector.run()


def _empty_cb(_metadata, _hfo_detector):
    return None


def _run_hfo_detection_with_plots_and_cb(plots, hfo_cb):
    run_hfo_detection_with_configuration(
        configuration=Configuration(
            data_path=get_hfo_directory('dummy'),
            measurement_mode=MeasurementMode.IEEG,
            hidden_neuron_count=86,
            plots=plots
        ),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=hfo_cb)


def test_channel_plotting_is_called_when_hfo_detector_is_called():
    with pytest.raises(ChannelDebugError):
        _run_hfo_detection_with_plots_and_cb(Plots(
            channel=[ChannelPlotKind.INTERNAL_CHANNEL_DEBUG],
            total=[]
        ), hfo_cb=_hfo_runner_cb)


def test_channel_plotting_is_not_called_when_hfo_detector_is_not_called():
    _run_hfo_detection_with_plots_and_cb(Plots(
        channel=[ChannelPlotKind.INTERNAL_CHANNEL_DEBUG],
        total=[]
    ), hfo_cb=_empty_cb)


def test_total_plotting_is_called():
    with pytest.raises(TotalDebugError):
        _run_hfo_detection_with_plots_and_cb(Plots(
            channel=[],
            total=[TotalPlotKind.INTERNAL_TOTAL_DEBUG]
        ), hfo_cb=_empty_cb)
