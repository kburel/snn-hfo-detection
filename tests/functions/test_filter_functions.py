from decimal import DivisionByZero
import pytest
from SNN_HFO_iEEG.Functions.Filter_functions import *


@pytest.mark.parametrize(
    "lowcut, highcut, fs, expected_coefficients",
    [(0.4, 0.4, 1, ([0, 0, 0, 0, 0, 0, 0, 0], [0]))]
)
def test_butter_bandpass(lowcut, highcut, fs, expected_coefficients):
    actual_a, actual_b = butter_bandpass(lowcut, highcut, fs)

    def to_np_array(arr): return np.array([pytest.approx(_) for _ in arr])
    expected_a, expected_b = to_np_array(expected_coefficients)
    print(expected_a)
    assert np.all(expected_a == actual_a)
    print(actual_b)
    assert np.all(expected_b == actual_b)


def test_butter_bandpass_raises_error_when_fs_is_zero():
    with pytest.raises(ZeroDivisionError):
        butter_bandpass(lowcut=1, highcut=1, fs=0)


def test_butter_bandpass_raises_error_when_cut_is_zero():
    with pytest.raises(ValueError):
        butter_bandpass(lowcut=0, highcut=1, fs=1)


_CUT_TO_FS_RATIO_LIMIT = 0.5


@pytest.mark.parametrize(
    "fs",
    [-1, -0.5, -0.1, 0.1, 0.5, 1]
)
def test_butter_bandpass_raises_error_when_cut_to_fs_ratio_is_too_big(fs):
    OUT_OF_BOUNDS_RATIO = _CUT_TO_FS_RATIO_LIMIT + 1e-3
    cut = fs * OUT_OF_BOUNDS_RATIO

    with pytest.raises(ValueError):
        butter_bandpass(lowcut=cut, highcut=cut, fs=fs)


@pytest.mark.parametrize(
    "fs",
    [-1, -0.5, -0.1, 0.1, 0.5, 1]
)
def test_butter_bandpass_passes_when_cut_to_fs_ratio_is_okay(fs):
    IN_BOUNDS_RATIO = _CUT_TO_FS_RATIO_LIMIT - 1e-3
    cut = fs * IN_BOUNDS_RATIO

    butter_bandpass(lowcut=cut, highcut=cut, fs=fs)
