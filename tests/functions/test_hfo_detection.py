from SNN_HFO_iEEG.Functions.HFO_detection_functions import *
import pytest


@pytest.mark.parametrize(
    "trial_duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection",
    [(0, [0], [0], 0.1, 0.1, {'total_HFO': 0,
      'time': [0], 'signal': [0], 'periods_HFO': [0, 0]}),
     (1, [0.5], [0.5], 0.1, 0.5, {'total_HFO': 0,
                                  'time': [0.5], 'signal': [1], 'periods_HFO': [0, 0]})]
)
def test_hfo_detection(trial_duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection):
    hfo_detection = detect_HFO(trial_duration, spike_monitor,
                               original_time_vector, step_size, window_size)
    print(hfo_detection)
    assert expected_hfo_detection['total_HFO'] == pytest.approx(
        hfo_detection['total_HFO'])
    assert np.all(expected_hfo_detection['time'] == [pytest.approx(_) for _ in
                                                     hfo_detection['time']])
    assert np.all(expected_hfo_detection['signal'] == [pytest.approx(_) for _ in
                                                       hfo_detection['signal']])
    assert np.all(expected_hfo_detection['periods_HFO'] == [pytest.approx(_) for _ in
                                                            hfo_detection['periods_HFO']])


def test_hfo_detection_fails_when_step_size_is_bigger_than_window():
    with pytest.raises(AssertionError):
        detect_HFO(0, [0], [0], 1, 0.5)


def test_hfo_detection_fails_when_step_size_is_zero():
    with pytest.raises(ZeroDivisionError):
        detect_HFO(0, [0], [0], 0, 0)
