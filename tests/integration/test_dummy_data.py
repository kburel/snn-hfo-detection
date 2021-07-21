
import os
from Run_Test_SNN_iEEG import run_hfo_detection
from pathlib import Path


def test_dummy_data():
    file_path = os.path.realpath(__file__)
    parent_dir = Path(file_path).parent.absolute()
    data_path = os.path.join(parent_dir, 'data', 'dummy')
    run_hfo_detection(data_path, lambda hfo_detection: print(hfo_detection))
