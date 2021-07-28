from typing import NamedTuple
import os
import numpy as np
import scipy.io as sio
from snn_hfo_ieeg.stages.loading.folder_discovery import get_patient_interval_paths


class PatientData(NamedTuple):
    '''
    Patient measurements
    '''
    wideband_signals: np.array
    signal_time: np.array


class ChannelData(NamedTuple):
    '''
    Patient measurements for a specific channel
    '''
    wideband_signal: np.array
    signal_time: np.array


def load_patient_data(full_intervals_path):
    interval = sio.loadmat(full_intervals_path)
    return PatientData(
        wideband_signals=interval['channels'],
        signal_time=interval['times'][0])


def extract_channel_data(patient_data, channel):
    return ChannelData(
        wideband_signal=patient_data.wideband_signals[channel],
        signal_time=patient_data.signal_time)
