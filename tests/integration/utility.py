import os
from snn_hfo_detection.entrypoint.hfo_detection import CustomOverrides
from tests.utility import get_tests_path

EMPTY_CUSTOM_OVERRIDES = CustomOverrides(
    duration=None,
    channels=None,
    intervals=None,
)


def get_hfo_directory(dataset_name):
    tests_path = get_tests_path()
    return os.path.join(tests_path, 'integration', 'data', dataset_name)
