import pytest
from snn_hfo_detection.functions.filter import *
from tests.utility import *

_CUT_TO_FS_RATIO_LIMIT = 0.5


@pytest.mark.parametrize(
    'lowcut, highcut, sampling_frequency, expected_coefficients',
    [(0.4, 0.4, 1, ([0., 0., 0., 0., 0., 0., 0., 0.,  0., 0., 0.], [1., 8.09016994, 31.18033989, 74.72135955, 122.81152949,
                                                                    144.35254916, 122.81152949, 74.72135955, 31.18033989, 8.09016994,
                                                                    1.])),
     (0.4, 0.3, 0.9, ([-0.0203585, -0., 0.10179248, -0., -0.20358497, -0.,
                       0.20358497, -0., -0.10179248, -0., 0.0203585], [1., 9.9849667, 47.59005786, 141.67098624, 290.88049757,
                                                                       429.8709108,  463.05607749, 359.26791433, 192.32998508, 64.1980211,
                                                                       10.15815918]))]
)
def test_butter_bandpass(lowcut, highcut, sampling_frequency, expected_coefficients):
    actual_a, actual_b = butter_bandpass(lowcut, highcut, sampling_frequency)
    expected_a, expected_b = expected_coefficients
    assert_are_lists_approximately_equal(actual_a, expected_a)
    assert_are_lists_approximately_equal(actual_b, expected_b)


def test_butter_bandpass_raises_error_when_fs_is_zero():
    with pytest.raises(ZeroDivisionError):
        butter_bandpass(lowcut=1, highcut=1, sampling_frequency=0)


def test_butter_bandpass_raises_error_when_cut_is_zero():
    with pytest.raises(ValueError):
        butter_bandpass(lowcut=0, highcut=1, sampling_frequency=1)


@pytest.mark.parametrize(
    'sampling_frequency',
    [-1, -0.5, -0.1, 0.1, 0.5, 1]
)
def test_butter_bandpass_raises_error_when_cut_to_sampling_frequency_ratio_is_too_big(sampling_frequency):
    out_of_bounds_ratio = _CUT_TO_FS_RATIO_LIMIT + 1e-3
    cut = sampling_frequency * out_of_bounds_ratio

    with pytest.raises(ValueError):
        butter_bandpass(lowcut=cut, highcut=cut,
                        sampling_frequency=sampling_frequency)


@pytest.mark.parametrize(
    'sampling_frequency',
    [-1, -0.5, -0.1, 0.1, 0.5, 1]
)
def test_butter_bandpass_passes_when_cut_to_sampling_frequency_ratio_is_okay(sampling_frequency):
    in_bounds_ratio = _CUT_TO_FS_RATIO_LIMIT - 1e-3
    cut = sampling_frequency * in_bounds_ratio

    butter_bandpass(lowcut=cut, highcut=cut,
                    sampling_frequency=sampling_frequency)


@pytest.mark.parametrize(
    'data, lowcut, highcut, sampling_frequency, expected_amplitude',
    [([0, 0, 0], 0.4, 0.4, 1, [0, 0, 0]),
     ([1, 2, 3], 0.4, 0.4, 1, [0, 0, 0]),
     ([0.5, 0.5, 0.5], 0.4, 0.4, 0.9, [0, 0, 0]),
     ([340, 1354, 50], 0.4, 0.2, 0.9,
      [-2140.50350966,  17514.88189302, -81064.75793357]),
     ([1, 1, -1], 0.2, 0.4, 0.9,
      [0.0332362,  -0.08709809, -0.08528257]),
     ]
)
def test_butter_bandpass_filter(data, lowcut, highcut, sampling_frequency, expected_amplitude):
    actual_amplitude = butter_bandpass_filter(
        data, lowcut, highcut, sampling_frequency)
    assert_are_lists_approximately_equal(actual_amplitude, expected_amplitude)
