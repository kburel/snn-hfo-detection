from typing import NamedTuple
import numpy as np
from snn_hfo_detection.stages.loading.patient_data import ChannelData
from snn_hfo_detection.user_facing_data import Bandwidth, FilteredSpikes, MeasurementMode
from snn_hfo_detection.functions.filter import butter_bandpass_filter
from snn_hfo_detection.functions.signal_to_spike.utility import find_thresholds, get_sampling_frequency, SignalToSpikeParameters
from snn_hfo_detection.functions.signal_to_spike.selector import signal_to_spike, SignalToSpikeAlgorithm


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
    signal_to_spike_algorithm: SignalToSpikeAlgorithm
        the underlying algorithm
    '''
    channel_data: ChannelData
    lowcut: int
    highcut: int
    scaling_factor: float
    calibration_time: float
    refractory_period: float
    signal_to_spike_algorithm: SignalToSpikeAlgorithm


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
    signal_to_spike_parameters = SignalToSpikeParameters(
        signal=signal,
        threshold_up=thresholds,
        threshold_down=thresholds,
        times=filter_parameters.channel_data.signal_time,
        refractory_period=filter_parameters.refractory_period,
        interpolation_factor=35_000
    )
    spike_trains = signal_to_spike(parameters=signal_to_spike_parameters,
                                   algorithm=filter_parameters.signal_to_spike_algorithm)
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
        refractory_period=3e-4,
        signal_to_spike_algorithm=configuration.signal_to_spike_algorithm,
    ))
    fast_ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=250,
        highcut=500,
        scaling_factor=scaling_factors.fast_ripple,
        calibration_time=configuration.calibration_time,
        refractory_period=3e-4,
        signal_to_spike_algorithm=configuration.signal_to_spike_algorithm,
    ))
    above_fast_ripple = _filter_signal_to_spike(_FilterParameters(
        channel_data=channel_data,
        lowcut=500,
        highcut=900,
        scaling_factor=scaling_factors.above_fast_ripple,
        calibration_time=configuration.calibration_time,
        refractory_period=1e-3,
        signal_to_spike_algorithm=configuration.signal_to_spike_algorithm,
    ))

    return FilteredSpikes(
        ripple=ripple,
        fast_ripple=fast_ripple,
        above_fast_ripple=above_fast_ripple)
