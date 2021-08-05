import pytest
from snn_hfo_ieeg.plotting.persistence import PlotMode
from snn_hfo_ieeg.user_facing_data import HfoDetection, Periods, Analytics, HfoDetectionWithAnalytics
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode
from snn_hfo_ieeg.entrypoint.hfo_detection import run_hfo_detection_with_configuration
from snn_hfo_ieeg.plotting.plot_loader import PlottingFunctions
from tests.utility import assert_are_hfo_detections_equal
from tests.integration.utility import get_hfo_directory, EMPTY_CUSTOM_OVERRIDES


# Empirical values
PERIOD_ACCURACY = 1.5
FREQUENCY_ACCURACY = 0.042


def _generate_test_configuration(dataset_name, measurement_mode=MeasurementMode.IEEG,):
    return Configuration(
        data_path=get_hfo_directory(dataset_name),
        measurement_mode=measurement_mode,
        hidden_neuron_count=86,
        calibration_time=10,
        plots=PlottingFunctions(
            channel=[],
            patient=[]
        ),
        disable_saving=True,
        saving_path=None,
        loading_path=None,
        plot_mode=PlotMode.SAVE,
        plot_path='plots/',
    )


def _assert_dummy_hfo_is_empty(hfo_detection_run):
    expected_hfo_detection = HfoDetectionWithAnalytics(
        result=HfoDetection(
            total_amount=0,
            frequency=0),
        analytics=Analytics(
            detections=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            periods=Periods(
                start=[],
                stop=[]
            ),
            # Not inspected
            filtered_spikes=None,
            spike_times=None,
            neuron_ids=None
        ))
    hfo_detection = hfo_detection_run.detector.run_with_analytics()
    assert_are_hfo_detections_equal(
        expected_hfo_detection, hfo_detection)


def test_dummy_data():
    run_hfo_detection_with_configuration(
        configuration=_generate_test_configuration('dummy'),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_assert_dummy_hfo_is_empty)


def _generate_add_detected_hfo_to_list_cb(detected_hfos):
    def add_detected_hfo_to_list(hfo_detection_run):
        hfo_detection_with_analytics = hfo_detection_run.detector.run_with_analytics()
        if hfo_detection_with_analytics.result.total_amount != 0:
            detected_hfos.append(hfo_detection_with_analytics)
    return add_detected_hfo_to_list


def _assert_contains_at_least(expected, actual, accuracy):
    approx_actual = [pytest.approx(element, abs=accuracy)
                     for element in actual]
    for element in expected:
        assert element in approx_actual


def test_ieeg_hfo_detection():
    detected_hfos = []
    run_hfo_detection_with_configuration(
        configuration=_generate_test_configuration('ieeg'),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo.result.frequency == pytest.approx(0.14, abs=FREQUENCY_ACCURACY)

    _assert_contains_at_least([0.0, 3.5, 6.43, 10.59, 14.29, 17.42],
                              hfo.analytics.periods.start, accuracy=PERIOD_ACCURACY)

    _assert_contains_at_least([0.06, 3.59, 6.54, 10.72, 14.39, 17.53],
                              hfo.analytics.periods.stop, accuracy=PERIOD_ACCURACY)


def test_ecog_hfo_detection():
    detected_hfos = []
    run_hfo_detection_with_configuration(
        configuration=_generate_test_configuration(
            'ecog', MeasurementMode.ECOG),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo.result.frequency == pytest.approx(0.07, abs=FREQUENCY_ACCURACY)

    _assert_contains_at_least([4.36, 9.85, 15.64, 35.5, 43.52, 53.64],
                              hfo.analytics.periods.start, accuracy=PERIOD_ACCURACY)

    _assert_contains_at_least([4.46, 9.94, 15.73, 36, 43.62, 53.73],
                              hfo.analytics.periods.stop, accuracy=PERIOD_ACCURACY)
