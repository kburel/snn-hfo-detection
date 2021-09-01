from typing import NamedTuple
import numpy as np
from snn_hfo_ieeg.stages.loading.patient_data import ChannelData
from snn_hfo_ieeg.functions.filter import butter_bandpass_filter
from snn_hfo_ieeg.functions.signal_to_spike import find_thresholds, signal_to_spike_refractory
from snn_hfo_ieeg.user_facing_data import MeasurementMode, Bandwidth, FilteredSpikes


SAMPLING_FREQUENCY = 2000


class _FilterParameters(NamedTuple):
    '''
    Parameters
    -------
    channel_data : ChannelData
        channel measurements
    lowcut: int
        lowcut frequency
    highcut: int
        highcut frequency
    scaling_factor: float
        new scaling factor
    calibration_time: float
        time that should be used to find thresholds
    '''
    channel_data: ChannelData
    lowcut: int
    highcut: int
    scaling_factor: float
    calibration_time: float


def _get_signal_times_in_calibration_time(signal, filter_parameters):
    signal_times = zip(signal, filter_parameters.channel_data.signal_time)
    signal_times_in_calibration = [(signal, time) for signal, time
                                   in signal_times
                                   if time <= filter_parameters.calibration_time]
    signals = np.array([signal for signal, _ in signal_times_in_calibration])
    times = np.array([time for _, time in signal_times_in_calibration])
    return signals, times


def _filter_signal_to_spike(filter_parameters) -> Bandwidth:
    signal = butter_bandpass_filter(data=filter_parameters.channel_data.wideband_signal,
                                    lowcut=filter_parameters.lowcut,
                                    highcut=filter_parameters.highcut,
                                    sampling_frequency=SAMPLING_FREQUENCY,
                                    order=2)

    calibration_signals, calibration_times = _get_signal_times_in_calibration_time(
        signal, filter_parameters)
    thresholds = np.ceil(find_thresholds(signals=calibration_signals,
                                         times=calibration_times,
                                         window_size=0.5,
                                         sample_ratio=1/6,
                                         scaling_factor=filter_parameters.scaling_factor))
    spike_trains = signal_to_spike_refractory(interpolation_factor=35000,
                                              times=filter_parameters.channel_data.signal_time,
                                              amplitude=signal,
                                              thr_up=thresholds, thr_dn=thresholds,
                                              refractory_period=3e-4)
    return Bandwidth(
        signal=signal,
        spike_trains=spike_trains
    )


def filter_stage(channel_data, configuration) -> FilteredSpikes:
    ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=80,
        highcut=250,
        scaling_factor=0.6,
        calibration_time=configuration.calibration_time
    ))
    fast_ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=250,
        highcut=500,
        scaling_factor=0.3,
        calibration_time=configuration.calibration_time
    ))
    very_fast_ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=500,
        highcut=900,
        scaling_factor=0.3,
        calibration_time=configuration.calibration_time
    ))

    return FilteredSpikes(
        ripple=ripple,
        fast_ripple=fast_ripple,
        very_fast_ripple=very_fast_ripple)
