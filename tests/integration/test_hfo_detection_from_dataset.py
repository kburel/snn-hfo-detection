import os
from pathlib import Path
import pytest
from run_test_snn_ieeg import run_hfo_detection_for_all_channels
from tests.utility import are_hfo_detections_equal


def _assert_dummy_hfo_is_empty(hfo_detection):
    expected_hfo_detection = {'total_hfo': 0, 'time':
                              [0., 0.0005, 0.001, 0.0015,
                               0.002, 0.0025, 0.003, 0.0035,
                               0.004, 0.0045],
                              'signal': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              'periods_hfo': [0, 0]}
    assert are_hfo_detections_equal(expected_hfo_detection, hfo_detection)


def _get_hfo_directory(dataset_name):
    file_path = os.path.realpath(__file__)
    parent_dir = Path(file_path).parent.absolute()
    return os.path.join(parent_dir, 'data', dataset_name)


def test_dummy_data():
    data_path = _get_hfo_directory('dummy')
    run_hfo_detection_for_all_channels(data_path, _assert_dummy_hfo_is_empty)


def _generate_add_detected_hfo_to_list_cb(list):
    return lambda hfo_detection: list.append(hfo_detection) if hfo_detection['total_hfo'] != 0 else None


def test_hfo_data():
    data_path = _get_hfo_directory('hfo')
    detected_hfos = []
    run_hfo_detection_for_all_channels(
        data_path, _generate_add_detected_hfo_to_list_cb(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo['total_hfo'] == 1
    assert hfo['periods_hfo'][0] == [pytest.approx(0)]
    assert hfo['periods_hfo'][1] == [pytest.approx(0.06)]
