import os
from run_test_snn_ieeg import run_hfo_detection
from pathlib import Path
from tests.utility import *


def _assert_dummy_hfo_is_empty(hfo_detection):
    expected_hfo_detection = {'total_hfo': 0, 'time': [0., 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035,
                                                       0.004, 0.0045], 'signal': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 'periods_hfo': [0, 0]}
    assert are_hfo_detections_equal(expected_hfo_detection, hfo_detection)


def test_dummy_data():
    file_path = os.path.realpath(__file__)
    parent_dir = Path(file_path).parent.absolute()
    data_path = os.path.join(parent_dir, 'data', 'dummy')
    run_hfo_detection(data_path, _assert_dummy_hfo_is_empty)
