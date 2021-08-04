from typing import NamedTuple
import numpy as np
import scipy.io as sio
import h5py
from scipy.io.matlab.mio5_utils import chars_to_strings


class PatientData(NamedTuple):
    '''
    Patient measurements
    '''
    wideband_signals: np.array
    signal_time: np.array
    channel_labels: np.array


class ChannelData(NamedTuple):
    '''
    Patient measurements for a specific channel
    '''
    wideband_signal: np.array
    signal_time: np.array


def load_patient_data(full_intervals_path):
    interval = sio.loadmat(full_intervals_path, chars_as_strings=True)
    return PatientData(
        wideband_signals=interval['channels'],
        signal_time=interval['times'][0],
        channel_labels=interval['channel_labels'])


def extract_channel_data(patient_data, channel):
    return ChannelData(
        wideband_signal=patient_data.wideband_signals[channel],
        signal_time=patient_data.signal_time)
