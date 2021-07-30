import os
import pytest
from snn_hfo_ieeg.user_facing_data import HfoDetection, Periods, Analytics, HfoDetectionWithAnalytics
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode
from snn_hfo_ieeg.stages.filter import FilteredSpikes
from snn_hfo_ieeg.entrypoint.hfo_detection import CustomOverrides, run_hfo_detection_with_configuration
from tests.utility import are_hfo_detections_equal, get_tests_path
EMPTY_CUSTOM_OVERRIDES = CustomOverrides(
    duration=None,
    channels=None,
    patients=None,
    intervals=None,
)

PERIOD_ACCURACY = 0.1
FREQUENCY_ACCURACY = 0.02


def _get_hfo_directory(dataset_name):
    tests_path = get_tests_path()
    return os.path.join(tests_path, 'integration', 'data', dataset_name)


def _generate_test_configuration(dataset_name, measurement_mode=MeasurementMode.IEEG,):
    return Configuration(
        data_path=_get_hfo_directory(dataset_name),
        measurement_mode=measurement_mode,
        hidden_neuron_count=86,
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
            ),
            filtered_spikes=FilteredSpikes(
                ripple=None, fast_ripple=None)  # Not inspected
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


def _assert_contains_at_least(actual, expected, accuracy):
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
    assert hfo.result.frequency == pytest.approx(0.12, abs=FREQUENCY_ACCURACY)

    _assert_contains_at_least([0.0, 3.5, 6.43, 10.59, 14.29, 17.42, 24.2],
                              hfo.analytics.periods.start, accuracy=PERIOD_ACCURACY)

    _assert_contains_at_least([0.06, 3.59, 6.54, 10.72, 14.39, 17.53, 24.29],
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

    _assert_contains_at_least([4.36, 9.85, 15.64, 36.13, 43.52, 53.64],
                              hfo.analytics.periods.start, accuracy=PERIOD_ACCURACY)

    _assert_contains_at_least([4.46, 9.94, 15.73, 36.22, 43.62, 53.73],
                              hfo.analytics.periods.stop, accuracy=PERIOD_ACCURACY)
