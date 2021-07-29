import os
from snn_hfo_ieeg.stages.plotting.total.plot_total_dummy import TotalDebugError
from snn_hfo_ieeg.stages.plotting.channel.plot_channel_dummy import ChannelDebugError
import pytest
import numpy as np
from snn_hfo_ieeg.functions.hfo_detection import HfoDetection, Periods, Analytics, HfoDetectionWithAnalytics
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode
from snn_hfo_ieeg.entrypoint.hfo_detection import CustomOverrides, run_hfo_detection_with_configuration
from snn_hfo_ieeg.stages.plotting.plot_factory import ChannelPlotKind, Plots, TotalPlotKind
from tests.utility import are_hfo_detections_equal, are_lists_approximately_equal, get_tests_path

EMPTY_CUSTOM_OVERRIDES = CustomOverrides(
    duration=None,
    channels=None,
    patients=None,
    intervals=None,
)


def _get_hfo_directory(dataset_name):
    tests_path = get_tests_path()
    return os.path.join(tests_path, 'integration', 'data', dataset_name)


def _generate_test_configuration(dataset_name, measurement_mode=MeasurementMode.IEEG,):
    return Configuration(
        data_path=_get_hfo_directory(dataset_name),
        measurement_mode=measurement_mode,
        hidden_neuron_count=86,
        plots=Plots(
            channel=[],
            total=[]
        )
    )


def _assert_dummy_hfo_is_empty(_metadata, hfo_detector):
    expected_hfo_detection = HfoDetectionWithAnalytics(
        result=HfoDetection(
            total_amount=0,
            frequency=0),
        analytics=Analytics(
            detections=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            analyzed_times=[0., 0.0005, 0.001, 0.0015,
                            0.002, 0.0025, 0.003, 0.0035,
                            0.004, 0.0045],
            periods=Periods(
                start=[],
                stop=[]
            )
        ))
    hfo_detection = hfo_detector.run_with_analytics()
    assert are_hfo_detections_equal(
        expected_hfo_detection, hfo_detection)


def test_dummy_data():
    run_hfo_detection_with_configuration(
        configuration=_generate_test_configuration('dummy'),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_assert_dummy_hfo_is_empty)


def _generate_add_detected_hfo_to_list_cb(detected_hfos):
    def add_detected_hfo_to_list(_metadata, hfo_detector):
        hfo_detection_with_analytics = hfo_detector.run_with_analytics()
        if hfo_detection_with_analytics.result.total_amount != 0:
            detected_hfos.append(hfo_detection_with_analytics)
    return add_detected_hfo_to_list


def test_iieg_hfo_detection():
    np.random.seed(0)

    detected_hfos = []
    run_hfo_detection_with_configuration(
        configuration=_generate_test_configuration('ieeg'),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo.result.total_amount == 1
    assert hfo.result.frequency == pytest.approx(0.01998021)
    assert hfo.analytics.periods.start == [pytest.approx(0)]
    assert hfo.analytics.periods.stop == [pytest.approx(0.0605)]

    np.random.seed(None)


def test_ecog_hfo_detection():
    np.random.seed(100)

    detected_hfos = []
    run_hfo_detection_with_configuration(
        configuration=_generate_test_configuration(
            'ecog', MeasurementMode.ECOG),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo.result.total_amount == 6
    assert hfo.result.frequency == pytest.approx(0.07994723482501549)

    assert are_lists_approximately_equal([4.36, 9.85, 15.64, 36.13, 43.52, 53.64],
                                         hfo.analytics.periods.start)

    assert are_lists_approximately_equal([4.4605, 9.9405, 15.7305, 36.2205, 43.6205, 53.73],
                                         hfo.analytics.periods.stop)

    np.random.seed(None)


def _hfo_runner_cb(_metadata, hfo_detector):
    hfo_detector.run()


def _empty_cb(_metadata, _hfo_detector):
    return None


def _run_hfo_detection_with_plots_and_cb(plots, hfo_cb):
    run_hfo_detection_with_configuration(
        configuration=Configuration(
            data_path=_get_hfo_directory("hfo"),
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
