import scipy.io as sio
from snn_hfo_detection.user_facing_data import PatientData, ChannelData


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
