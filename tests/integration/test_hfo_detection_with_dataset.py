import pytest
import os
from Run_Test_SNN_iEEG import run_hfo_detection
from pathlib import Path

def _assert_dummy_hfo_is_empty(hfo_detection):
    expected_hfo_detection = {'total_HFO': 0, 'time': [0.    , 0.0005, 0.001 , 0.0015, 0.002 , 0.0025, 0.003 , 0.0035,
       0.004 , 0.0045], 'signal': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 'periods_HFO': [0, 0]}
    print(f'hfo_detection: {hfo_detection}')

def test_dummy_data():
    file_path = os.path.realpath(__file__)
    parent_dir = Path(file_path).parent.absolute()
    data_path = os.path.join(parent_dir, 'data', 'dummy')
    run_hfo_detection(data_path, _assert_dummy_hfo_is_empty)
    pytest.fail()