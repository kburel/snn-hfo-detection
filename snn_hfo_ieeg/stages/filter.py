from typing import NamedTuple
import numpy as np
from snn_hfo_ieeg.stages.loading.patient_data import ChannelData
from snn_hfo_ieeg.functions.filter import butter_bandpass_filter
from snn_hfo_ieeg.functions.signal_to_spike import find_thresholds, signal_to_spike_refractory


SAMPLING_FREQUENCY = 2000


class Ripple(NamedTuple):
    '''
    Up and down spikes in a ripple bandwidth (80-250Hz)
    '''
    up: np.array
    down: np.array


class FastRipple(NamedTuple):
    '''
    Up and down spikes in a fast ripple bandwidth (250-500Hz)
    '''
    up: np.array
    down: np.array


class FilteredSpikes(NamedTuple):
    '''
    Spikes filtered in the ripple and fast ripple bandwidths
    '''
    ripple: Ripple
    fast_ripple: FastRipple


class FilterParameters(NamedTuple):
    '''
    Parameters
    -------
    channel_data: ChannelData
        channel measurements
    adm_parameters: dict
        loaded adm parameters from .mat file
    lowcut: int
        lowcut frequency
    highcut: int
        highcut frequency
    scaling_factor: float
        new scaling factor
    '''
    channel_data: ChannelData
    adm_parameters: dict
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
    return signal_to_spike_refractory(interpfact=filter_parameters.adm_parameters['interpfact'][0][0],
                                      time=filter_parameters.channel_data.signal_time,
                                      amplitude=signal,
                                      thr_up=threshold, thr_dn=threshold,
                                      refractory_period=filter_parameters.adm_parameters['refractory'][0][0])


def filter_stage(channel_data, adm_parameters):
    r_filter_parameters = FilterParameters(
        channel_data=channel_data,
        adm_parameters=adm_parameters,
        lowcut=80,
        highcut=250,
        scaling_factor=adm_parameters['Ripple_sf'][0][0]
    )

    fr_filter_parameters = FilterParameters(
        channel_data=channel_data,
        adm_parameters=adm_parameters,
        lowcut=250,
        highcut=500,
        scaling_factor=adm_parameters['FR_sf'][0][0]
    )

    r_up, r_dn = _filter_signal_to_spike(r_filter_parameters)
    fr_up, fr_dn = _filter_signal_to_spike(fr_filter_parameters)

    ripple = Ripple(up=r_up, down=r_dn)
    fast_ripple = FastRipple(up=fr_up, down=fr_dn)

    return FilteredSpikes(ripple=ripple, fast_ripple=fast_ripple)
