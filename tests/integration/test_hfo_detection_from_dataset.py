import pytest
from snn_hfo_detection.plotting.persistence import PlotMode
from snn_hfo_detection.user_facing_data import HfoDetection, Periods, Analytics, HfoDetectionWithAnalytics
from snn_hfo_detection.user_facing_data import Configuration, MeasurementMode
from snn_hfo_detection.entrypoint.hfo_detection import run_hfo_detection_with_configuration
from snn_hfo_detection.plotting.plot_loader import PlottingFunctions
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


@pytest.mark.parametrize(
    'dataset, measurement_mode, frequency',
    [('ieeg', MeasurementMode.IEEG, 0.08),
     ('ecog', MeasurementMode.ECOG, 0.0266),
     ('scalp', MeasurementMode.SCALP, 0.04)]
)
def test_hfo_detection_frequency(dataset, measurement_mode, frequency):
    detected_hfos = []
    run_hfo_detection_with_configuration(
        configuration=_generate_test_configuration(dataset, measurement_mode),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo.result.frequency == pytest.approx(
        frequency, abs=FREQUENCY_ACCURACY)
