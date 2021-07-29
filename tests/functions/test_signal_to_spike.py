import pytest
from snn_hfo_ieeg.functions.signal_to_spike import *
from tests.utility import *


@pytest.mark.parametrize(
    'signals, time, window, step_size, sample_ratio, scaling_factor, expected_mean_threshold',
    [(np.array([0, 1]),
      np.array([0, 1]), 1, 1, 0.9, 0.1, 0.1),
     (np.array([-0.6, -2, -5, 10, 20, 0.2, -3, 0.4]),
      np.arange(0, 0.8, 0.1), 1, 1, 0.9, 0.6, 15.0),
     (np.arange(-20, 20, 0.1),
      np.arange(0, 4, 0.01), 1, 1, 0.4, 0.6, -6)]
)
def test_find_thresholds(signals, time, window, step_size, sample_ratio, scaling_factor, expected_mean_threshold):
    mean_threshold = find_thresholds(
        signals, time, window, step_size, sample_ratio, scaling_factor)
    assert expected_mean_threshold == pytest.approx(mean_threshold)


@pytest.mark.parametrize(
    'sample_ratio',
    [-1, 0, 1, 2]
)
def test_find_thresholds_does_not_accept_invalid_percentages(sample_ratio):
    with pytest.raises(ValueError):
        find_thresholds(
            signals=np.array([0, 1]),
            times=np.array([0, 1]),
            window_size=1,
            step_size=1,
            sample_ratio=sample_ratio,
            scaling_factor=0.1)


@pytest.mark.parametrize(
    'interpfact, time, amplitude, thr_up, thr_dn, refractory_period, expected_spike_trains',
    [(1, [0, 1], [1, 0], 3, 0.5, 0.01, SpikeTrains(up=[], down=[])),
     (10, [0, 1, 2], [0, 10, -20], 3, 0.5, 0.01, SpikeTrains(up=[0.3157894736842105, 0.631578947368421, 0.9473684210526315],
                                                             down=[1.0526315789473684, 1.1578947368421053,
                                                                   1.263157894736842, 1.3684210526315788, 1.4736842105263157,
                                                                   1.5789473684210527, 1.6842105263157894, 1.789473684210526,
                                                                   1.894736842105263, 2.0])),
     (100, np.arange(0, 4, 0.01), np.arange(-20, 20, 0.1), 3, -5, 0.6, SpikeTrains(up=[0.6015075376884422, 1.2030150753768845,
                                                                                       1.8045226130653267, 2.406030150753769,
                                                                                       3.007537688442211, 3.6090452261306534],
                                                                                   down=[0.0]))]
)
def test_signal_to_spike_refractory(interpfact, time, amplitude, thr_up, thr_dn, refractory_period, expected_spike_trains):
    spike_trains = signal_to_spike_refractory(
        interpfact, time, amplitude, thr_up, thr_dn, refractory_period)
    assert are_lists_approximately_equal(
        spike_trains.up, expected_spike_trains.up)
    assert are_lists_approximately_equal(
        spike_trains.down, expected_spike_trains.down)


@pytest.mark.parametrize(
    'spikes, expected_concatenation',
    [([np.array([1, 2, 3])], ([1, 2, 3], [0, 0, 0])),
     ([np.array([1, 2, 3]), np.array([4, 5, 6])], ([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1]))])
def test_concatenate_spikes(spikes, expected_concatenation):
    spike_times, neuron_ids = concatenate_spikes(spikes)
    expected_spike_times, expected_neuron_ids = expected_concatenation
    assert are_lists_approximately_equal(spike_times, expected_spike_times)
    assert are_lists_approximately_equal(neuron_ids, expected_neuron_ids)
