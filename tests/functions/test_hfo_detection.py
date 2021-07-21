from SNN_HFO_iEEG.Functions.HFO_detection_functions import *
import pytest
from tests.utility import *

@pytest.mark.parametrize(
    'trial_duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection',
    [(0, [0], [0], 0.1, 0.1, {'total_HFO': 0,
      'time': [0], 'signal': [0], 'periods_HFO': [0, 0]}),
     (1, [0.5], [0.5], 0.1, 0.5, {'total_HFO': 0,
                                  'time': [0.5], 'signal': [1], 'periods_HFO': [0, 0]}),
     (-1, [-0.5], [-0.5], -0.1, -0.05, {'total_HFO': 0,
                                        'time': [-0.5], 'signal': [0], 'periods_HFO': [0, 0]}),
     (0.001, np.arange(0, 300, 5e-4), [0.2, 0.3], 0.01, 0.05, {'total_HFO': 0,
                                                               'time': [0.2, 0.3], 'signal': [0, 0], 'periods_HFO': [0, 0]})]
)
def test_hfo_detection(trial_duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection):
    hfo_detection = detect_HFO(trial_duration, spike_monitor,
                               original_time_vector, step_size, window_size)
    assert are_hfo_detections_equal(expected_hfo_detection, hfo_detection)

def test_hfo_detection_fails_when_step_size_is_bigger_than_window():
    with pytest.raises(AssertionError):
        detect_HFO(0, [0], [0], 1, 0.5)


def test_hfo_detection_fails_when_step_size_is_zero():
    with pytest.raises(ZeroDivisionError):
        detect_HFO(0, [0], [0], 0, 0)
