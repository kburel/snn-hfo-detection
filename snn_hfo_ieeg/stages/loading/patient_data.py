from typing import NamedTuple
import os
import numpy as np
import scipy.io as sio


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


def load_patient_data(patient, interval, data_path):
    file_name = f'P{patient}/I{interval}.mat'
    interval = sio.loadmat(os.path.join(data_path, file_name))
    return PatientData(
        wideband_signals=interval['chb'],
        signal_time=interval['t'][0])


def extract_channel_data(patient_data, channel):
    return ChannelData(
        wideband_signal=patient_data.wideband_signals[channel],
        signal_time=patient_data.signal_time)
