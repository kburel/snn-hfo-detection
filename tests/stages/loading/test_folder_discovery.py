import os
from tests.utility import get_tests_path
from snn_hfo_ieeg.stages.loading.folder_discovery import Patients, get_patient_interval_paths


def test_filters_patient_directories():
    tests_path = get_tests_path()
    test_data_path = os.path.join(tests_path, 'stages', 'loading', 'test_data')
    patients = get_patient_interval_paths(test_data_path)
    expected_patients: Patients = {
        0: {
            0: os.path.join(test_data_path, 'P0', 'I0.mat'),
        },
        1: {
            1: os.path.join(test_data_path, 'P1', 'I1.mat'),
            2: os.path.join(test_data_path, 'P1', 'I2.mat'),
        },
        2: {
            1: os.path.join(test_data_path, 'P2', 'I1.mat'),
        },
        4: {
            5: os.path.join(test_data_path, 'P4', 'I5.mat'),
        }
    }
    assert expected_patients == patients
