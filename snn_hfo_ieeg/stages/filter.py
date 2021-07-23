from typing import NamedTuple
import numpy as np
from snn_hfo_ieeg.functions.filter import butter_bandpass_filter
from snn_hfo_ieeg.functions.signal_to_spike import find_thresholds, signal_to_spike_refractory


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
    wideband_signal: np.array
        array of measures signals
    signal_time: np.array
        array of measured times
    adm_parameters: dict
        loaded adm parameters from .mat file
    lowcut: int
        lowcut frequency
    highcut: int
        highcut frequency
    sampling_frequency: int
        new sampling frequency
    scaling_factor: float
        new scaling factor
    '''
    wideband_signal: np.array
    signal_time: np.array
    adm_parameters: dict
    lowcut: int
    highcut: int
    sampling_frequency: int
    scaling_factor: float


def _filter_signal_to_spike(filter_parameters):
    signal = butter_bandpass_filter(data=filter_parameters.wideband_signal,
                                    lowcut=filter_parameters.lowcut,
                                    highcut=filter_parameters.highcut,
                                    sampling_frequency=filter_parameters.sampling_frequency,
                                    order=2)
    threshold = np.ceil(find_thresholds(signal=signal,
                                        time=filter_parameters.signal_time,
                                        window=1,
                                        step_size=1,
                                        chosen_samples=50,
                                        scaling_factor=filter_parameters.scaling_factor))
    return signal_to_spike_refractory(interpfact=filter_parameters.adm_parameters['interpfact'][0][0],
                                      time=filter_parameters.signal_time,
                                      amplitude=signal,
                                      thr_up=threshold, thr_dn=threshold,
                                      refractory_period=filter_parameters.adm_parameters['refractory'][0][0])


def filter_stage(wideband_signal, sampling_frequency, signal_time, adm_parameters):
    r_filter_parameters = FilterParameters(
        wideband_signal=wideband_signal,
        signal_time=signal_time,
        sampling_frequency=sampling_frequency,
        adm_parameters=adm_parameters,
        lowcut=80,
        highcut=250,
        scaling_factor=adm_parameters['Ripple_sf'][0][0]
    )

    fr_filter_parameters = FilterParameters(
        wideband_signal=wideband_signal,
        signal_time=signal_time,
        sampling_frequency=sampling_frequency,
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
