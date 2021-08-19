
import pytest
from snn_hfo_ieeg.plotting.persistence import PlotMode
from snn_hfo_ieeg.plotting.plot_loader import find_plotting_functions
from snn_hfo_ieeg.user_facing_data import Configuration, MeasurementMode
from snn_hfo_ieeg.entrypoint.hfo_detection import run_hfo_detection_with_configuration
from snn_hfo_ieeg.plotting.plot_channel import ChannelDebugError
from snn_hfo_ieeg.plotting.plot_patient import PatientDebugError
from tests.integration.utility import get_hfo_directory, EMPTY_CUSTOM_OVERRIDES


def _hfo_runner_cb(_hfo_detection_run):
    _hfo_detection_run.detector.run()


def _empty_cb(_hfo_detection_run):
    return None


def _run_hfo_detection_with_plot_and_cb(plot_name, hfo_cb):
    run_hfo_detection_with_configuration(
        configuration=Configuration(
            data_path=get_hfo_directory('dummy'),
            measurement_mode=MeasurementMode.IEEG,
            hidden_neuron_count=86,
            plots=find_plotting_functions([plot_name]),
            calibration_time=10,
            saving_path=None,
            disable_saving=True,
            loading_path=None,
            plot_mode=PlotMode.SAVE,
            plot_path='plots/',
        ),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=hfo_cb)


def test_channel_plotting_is_called_when_hfo_detector_is_called():
    with pytest.raises(ChannelDebugError):
        _run_hfo_detection_with_plot_and_cb(
            'internal_channel_debug', hfo_cb=_hfo_runner_cb)


def test_channel_plotting_is_not_called_when_hfo_detector_is_not_called():
    _run_hfo_detection_with_plot_and_cb(
        'internal_channel_debug', hfo_cb=_empty_cb)


def test_patient_plotting_is_called():
    with pytest.raises(PatientDebugError):
        _run_hfo_detection_with_plot_and_cb(
            'internal_patient_debug', hfo_cb=_empty_cb)
