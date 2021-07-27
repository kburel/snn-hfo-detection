import os
from pathlib import Path
import pytest
from snn_hfo_ieeg.functions.hfo_detection import HfoDetection, Periods, PlottingData
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode
from run_test_snn_ieeg import CustomOverrides, run_hfo_detection_for_all_channels
from tests.utility import are_hfo_detections_equal

EMPTY_CUSTOM_OVERRIDES = CustomOverrides(
    duration=None,
    channels=None
)


def _get_hfo_directory(dataset_name):
    file_path = os.path.realpath(__file__)
    parent_dir = Path(file_path).parent.absolute()
    return os.path.join(parent_dir, 'data', dataset_name)


def _generate_test_configuration(dataset_name, measurement_mode=MeasurementMode.IEEG,):
    return Configuration(
        data_path=_get_hfo_directory(dataset_name),
        measurement_mode=measurement_mode,
        hidden_neuron_count=86,
    )


def _assert_dummy_hfo_is_empty(hfo_detection):
    expected_hfo_detection = HfoDetection(
        total_amount=0,
        frequency=0,
        plotting_data=PlottingData(
            detections=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            analyzed_times=[0., 0.0005, 0.001, 0.0015,
                            0.002, 0.0025, 0.003, 0.0035,
                            0.004, 0.0045],
            periods=Periods(
                start=[],
                stop=[]
            )
        ))
    assert are_hfo_detections_equal(expected_hfo_detection, hfo_detection)


def test_dummy_data():
    run_hfo_detection_for_all_channels(
        configuration=_generate_test_configuration('dummy'),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_assert_dummy_hfo_is_empty)


def _generate_add_detected_hfo_to_list_cb(list):
    return lambda hfo_detection: list.append(hfo_detection) if hfo_detection.total_amount != 0 else None


def test_iieg_hfo_detection():
    detected_hfos = []
    run_hfo_detection_for_all_channels(
        configuration=_generate_test_configuration('ieeg'),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo.total_amount == 1
    assert hfo.frequency == pytest.approx(0.019980219582613215)
    assert hfo.plotting_data.periods.start == [pytest.approx(0)]
    assert hfo.plotting_data.periods.stop == [pytest.approx(0.0605)]


def test_ecog_hfo_detection():
    detected_hfos = []
    run_hfo_detection_for_all_channels(
        configuration=_generate_test_configuration(
            'ecog', MeasurementMode.ECOG),
        custom_overrides=EMPTY_CUSTOM_OVERRIDES,
        hfo_cb=_generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo.total_amount == 1
    assert hfo.frequency == pytest.approx(0.019980219582613215)
    assert hfo.plotting_data.periods.start == [pytest.approx(4.37)]
    assert hfo.plotting_data.periods.stop == [pytest.approx(4.41)]
