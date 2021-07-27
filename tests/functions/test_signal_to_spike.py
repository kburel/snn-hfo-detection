import pytest
from snn_hfo_ieeg.functions.signal_to_spike import *
from tests.utility import *


@pytest.mark.parametrize(
    'signal, time, window, step_size, chosen_samples, scaling_factor, expected_mean_threshold',
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
    assert expected_mean_threshold == pytest.approx(mean_threshold)


@pytest.mark.parametrize(
    'interpfact, time, amplitude, thr_up, thr_dn, refractory_period, expected_thresholds',
    [(1, [0, 1], [1, 0], 3, 0.5, 0.01, ([], [])),
     (10, [0, 1, 2], [0, 10, -20], 3, 0.5, 0.01, ([0.3157894736842105, 0.631578947368421, 0.9473684210526315], [1.0526315789473684, 1.1578947368421053,
                                                                                                                1.263157894736842, 1.3684210526315788, 1.4736842105263157, 1.5789473684210527, 1.6842105263157894, 1.789473684210526, 1.894736842105263, 2.0])),
     (100, np.arange(0, 4, 0.01), np.arange(-20, 20, 0.1), 3, -5, 0.6, ([0.6015075376884422,
                                                                         1.2030150753768845, 1.8045226130653267, 2.406030150753769, 3.007537688442211, 3.6090452261306534], [0.0]))]
)
def test_signal_to_spike_refractory(interpfact, time, amplitude, thr_up, thr_dn, refractory_period, expected_thresholds):
    spike_up, spike_dn = signal_to_spike_refractory(
        interpfact, time, amplitude, thr_up, thr_dn, refractory_period)
    expected_spike_up, expected_spike_dn = expected_thresholds
    assert are_lists_approximately_equal(spike_up, expected_spike_up)
    assert are_lists_approximately_equal(spike_dn, expected_spike_dn)


@pytest.mark.parametrize(
    'spikes, expected_concatenation',
    [([np.array([1, 2, 3])], ([1, 2, 3], [0, 0, 0])),
     ([np.array([1, 2, 3]), np.array([4, 5, 6])], ([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1]))])
def test_concatenate_spikes(spikes, expected_concatenation):
    spike_times, neuron_ids = concatenate_spikes(spikes)
    expected_spike_times, expected_neuron_ids = expected_concatenation
    assert are_lists_approximately_equal(spike_times, expected_spike_times)
    assert are_lists_approximately_equal(neuron_ids, expected_neuron_ids)
