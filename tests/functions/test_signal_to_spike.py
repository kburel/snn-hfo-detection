from SNN_HFO_iEEG.Functions.Signal_to_spike_functions import *
import pytest


@pytest.mark.parametrize(
    "signal, time, window, step_size, chosen_samples, scaling_factor, expected_mean_threshold",
    [(np.array([0, 1]),
      np.array([0, 1]), 1, 1, 1, 0.1, 0.1),
     (np.array([-0.6, -2, -5, 10, 20, 0, -3, 0.4]),
      np.arange(0, 0.8, 0.1), 1, 1, 50, 0.6, 15.0),
     (np.arange(-20, 20, 0.1),
      np.arange(0, 4, 0.01), 1, 1, 50, 0.6, 5.985)]
)
def test_find_thresholds(signal, time, window, step_size, chosen_samples, scaling_factor, expected_mean_threshold):
    mean_threshold = find_thresholds(
        signal, time, window, step_size, chosen_samples, scaling_factor)
    print(mean_threshold)
    assert expected_mean_threshold == pytest.approx(mean_threshold)
