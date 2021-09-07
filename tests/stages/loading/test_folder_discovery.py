import os
import pytest
from tests.utility import get_tests_path
from snn_hfo_detection.stages.loading.folder_discovery import get_interval_paths


def _get_test_data_path():
    tests_path = get_tests_path()
    return os.path.join(
        tests_path, 'stages', 'loading', 'test_data')


def _get_patient_path(patient):
    test_data_path = _get_test_data_path()
    return os.path.join(test_data_path, f'P{patient}')


def _get_interval_path(patient, interval):
    patient_path = _get_patient_path(patient)
    return os.path.join(patient_path, f'I{interval}.mat')


@pytest.mark.parametrize(
    'input_path, expected_intervals',
    [(_get_patient_path(0), {0: _get_interval_path(0, 0)}),
     (_get_patient_path(1), {1: _get_interval_path(1, 1),
                             2: _get_interval_path(1, 2)}),
     (_get_patient_path(2), {1: _get_interval_path(2, 1)}),
     (_get_patient_path(3), {5: _get_interval_path(3, 5)}),
     (_get_patient_path(4), {}),
     (_get_test_data_path(), {})]
)
def test_filters_patient_directories(input_path, expected_intervals):
    intervals = get_interval_paths(input_path)
    assert expected_intervals == intervals
