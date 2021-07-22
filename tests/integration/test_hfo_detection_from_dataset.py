import os
from Run_Test_SNN_iEEG import run_hfo_detection
from pathlib import Path
from tests.utility import *


def _assert_dummy_hfo_is_empty(hfo_detection):
    expected_hfo_detection = {'total_HFO': 0, 'time': [0., 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035,
                                                       0.004, 0.0045], 'signal': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 'periods_HFO': [0, 0]}
    assert are_hfo_detections_equal(expected_hfo_detection, hfo_detection)


def _get_hfo_directory(dataset_name):
    file_path = os.path.realpath(__file__)
    parent_dir = Path(file_path).parent.absolute()
    return os.path.join(parent_dir, 'data', dataset_name)


def test_dummy_data():
    data_path = _get_hfo_directory('dummy')
    run_hfo_detection(data_path, _assert_dummy_hfo_is_empty)


def _generate_add_detected_hfo_to_list_callback(list):
    return lambda hfo_detection: list.append(hfo_detection) if hfo_detection['total_HFO'] != 0 else None


def test_hfo_data():
    data_path = _get_hfo_directory('hfo')
    detected_hfos = []
    run_hfo_detection(
        data_path, _generate_add_detected_hfo_to_list_callback(detected_hfos))
    assert len(detected_hfos) == 1
    hfo = detected_hfos[0]
    assert hfo['total_HFO'] == 1
    assert hfo['periods_HFO'][0] == [pytest.approx(0)]
    assert hfo['periods_HFO'][1] == [pytest.approx(0.06)]
