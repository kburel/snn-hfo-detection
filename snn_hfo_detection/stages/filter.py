from typing import NamedTuple
import numpy as np
from snn_hfo_detection.stages.loading.patient_data import ChannelData
from snn_hfo_detection.functions.filter import butter_bandpass_filter
from snn_hfo_detection.functions.signal_to_spike import find_thresholds, signal_to_spike, get_sampling_frequency
from snn_hfo_detection.user_facing_data import Bandwidth, FilteredSpikes, MeasurementMode


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
    refractory_period: float
        time for the refractory period
    '''
    channel_data: ChannelData
    lowcut: int
    highcut: int
    scaling_factor: float
    calibration_time: float
    refractory_period: float


def _get_signal_times_in_calibration_time(signal, filter_parameters):
    signal_times = zip(signal, filter_parameters.channel_data.signal_time)
    signal_times_in_calibration = [(signal, time) for signal, time
                                   in signal_times
                                   if time <= filter_parameters.calibration_time]
    signals = np.array([signal for signal, _ in signal_times_in_calibration])
    times = np.array([time for _, time in signal_times_in_calibration])
    return signals, times


def _filter_signal_to_spike(filter_parameters: _FilterParameters) -> Bandwidth:
    sampling_frequency = get_sampling_frequency(
        filter_parameters.channel_data.signal_time)
    signal = butter_bandpass_filter(data=filter_parameters.channel_data.wideband_signal,
                                    lowcut=filter_parameters.lowcut,
                                    highcut=filter_parameters.highcut,
                                    sampling_frequency=sampling_frequency,
                                    order=2)

    calibration_signals, calibration_times = _get_signal_times_in_calibration_time(
        signal, filter_parameters)
    thresholds = np.ceil(find_thresholds(signals=calibration_signals,
                                         times=calibration_times,
                                         window_size=0.5,
                                         sample_ratio=1/6,
                                         scaling_factor=filter_parameters.scaling_factor))
    spike_trains = signal_to_spike(input_signal=signal,
                                   threshold_up=thresholds,
                                   threshold_down=thresholds,
                                   times=filter_parameters.channel_data.signal_time,
                                   refractory_period_duration=filter_parameters.refractory_period)
    return Bandwidth(
        signal=signal,
        spike_trains=spike_trains
    )


class _ScalingFactors(NamedTuple):
    ripple: float
    fast_ripple: float
    above_fast_ripple: float


def _get_scaling_factors(configuration):
    base_factors = _ScalingFactors(
        ripple=0.6,
        fast_ripple=0.3,
        above_fast_ripple=0.3)
    if configuration.measurement_mode is MeasurementMode.IEEG:
        return _ScalingFactors(
            ripple=base_factors.ripple,
            fast_ripple=base_factors.fast_ripple,
            above_fast_ripple=base_factors.above_fast_ripple)
    if configuration.measurement_mode is MeasurementMode.ECOG:
        return _ScalingFactors(
            ripple=base_factors.ripple,
            fast_ripple=0.5,
            above_fast_ripple=base_factors.above_fast_ripple)
    if configuration.measurement_mode is MeasurementMode.SCALP:
        return _ScalingFactors(
            ripple=0.3,
            fast_ripple=base_factors.fast_ripple,
            above_fast_ripple=base_factors.above_fast_ripple)
    raise ValueError("Unknown measurement mode")


def filter_stage(channel_data, configuration) -> FilteredSpikes:
    scaling_factors = _get_scaling_factors(configuration)
    ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=80,
        highcut=250,
        scaling_factor=scaling_factors.ripple,
        calibration_time=configuration.calibration_time,
        refractory_period=3e-4
    ))
    fast_ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=250,
        highcut=500,
        scaling_factor=scaling_factors.fast_ripple,
        calibration_time=configuration.calibration_time,
        refractory_period=3e-4
    ))
    above_fast_ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=500,
        highcut=900,
        scaling_factor=scaling_factors.above_fast_ripple,
        calibration_time=configuration.calibration_time,
        refractory_period=1e-3
    ))

    return FilteredSpikes(
        ripple=ripple,
        fast_ripple=fast_ripple,
        above_fast_ripple=above_fast_ripple)
