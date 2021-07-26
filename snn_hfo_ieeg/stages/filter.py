from typing import NamedTuple
import numpy as np
from snn_hfo_ieeg.stages.loading.patient_data import ChannelData
from snn_hfo_ieeg.functions.filter import butter_bandpass_filter
from snn_hfo_ieeg.functions.signal_to_spike import find_thresholds, signal_to_spike_refractory
from snn_hfo_ieeg.stages.shared_config import MeasurementMode


SAMPLING_FREQUENCY = 2000


class Bandwidth(NamedTuple):
    '''
    Up and down spikes in a bandwidth (80-250Hz)
    '''
    up: np.array
    down: np.array


class FilteredSpikes(NamedTuple):
    '''
    Spikes filtered in the ripple and fast ripple bandwidths

    Parameters
    -----
    ripple : Bandwidth
        signals in ripple bandwidth (80-250Hz)
    fast_ripple : Bandwidth
        signals in fast ripple bandwidth (250-500Hz)
    '''
    ripple: Bandwidth
    fast_ripple: Bandwidth


class FilterParameters(NamedTuple):
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
    '''
    channel_data: ChannelData
    lowcut: int
    highcut: int
    scaling_factor: float


def _filter_signal_to_spike(filter_parameters):
    signal = butter_bandpass_filter(data=filter_parameters.channel_data.wideband_signal,
                                    lowcut=filter_parameters.lowcut,
                                    highcut=filter_parameters.highcut,
                                    sampling_frequency=SAMPLING_FREQUENCY,
                                    order=2)
    threshold = np.ceil(find_thresholds(signal=signal,
                                        time=filter_parameters.channel_data.signal_time,
                                        window=1,
                                        step_size=1,
                                        chosen_samples=50,
                                        scaling_factor=filter_parameters.scaling_factor))
    return signal_to_spike_refractory(interpfact=35000,
                                      time=filter_parameters.channel_data.signal_time,
                                      amplitude=signal,
                                      thr_up=threshold, thr_dn=threshold,
                                      refractory_period=3e-4)


def filter_stage(channel_data, configuration):
    r_filter_parameters = FilterParameters(
        channel_data=channel_data,
        lowcut=80,
        highcut=250,
        scaling_factor=0.6
    )

    fr_filter_parameters = FilterParameters(
        channel_data=channel_data,
        lowcut=250,
        highcut=500,
        scaling_factor=0.3
    )

    r_up, r_dn = _filter_signal_to_spike(r_filter_parameters)
    fr_up, fr_dn = _filter_signal_to_spike(fr_filter_parameters)

    ripple = Bandwidth(up=r_up, down=r_dn)
    fast_ripple = Bandwidth(up=fr_up, down=fr_dn)

    if configuration.measurement_mode is MeasurementMode.IEEG:
        return FilteredSpikes(ripple=ripple, fast_ripple=fast_ripple)
    elif configuration.measurement_mode is MeasurementMode.ECOG:
        return FilteredSpikes(ripple=None, fast_ripple=fast_ripple)
    elif configuration.measurement_mode is MeasurementMode.SCALP:
        return FilteredSpikes(ripple=ripple, fast_ripple=None)
    raise ValueError(
        f'configuration.measurement_mode has an invalid value. Allowed values: {MeasurementMode}, instead got: {configuration.measurement_mode}')
