from typing import NamedTuple
import numpy as np
from snn_hfo_ieeg.functions.filter import butter_bandpass_filter
from snn_hfo_ieeg.functions.signal_to_spike import find_thresholds, signal_to_spike_refractory


class Ripple(NamedTuple):
    up: np.array
    down: np.array


class FastRipple(NamedTuple):
    up: np.array
    down: np.array


class FilteredSpikes(NamedTuple):
    ripple: Ripple
    fast_ripple: FastRipple


def filter_stage(wideband_signal, sampling_frequency, signal_time, adm_parameters):
    # Filter the Wideband in ripple and fr bands
    r_signal = butter_bandpass_filter(data=wideband_signal,
                                      lowcut=80,
                                      highcut=250,
                                      sampling_frequency=sampling_frequency,
                                      order=2)
    fr_signal = butter_bandpass_filter(data=wideband_signal,
                                       lowcut=250,
                                       highcut=500,
                                       sampling_frequency=sampling_frequency,
                                       order=2)

    # ==================================
    # Baseline detection stage
    # ==================================
    # Based on the noise floor find the signal-to-spike thresholds
    r_threshold = np.ceil(find_thresholds(signal=r_signal,
                                          time=signal_time,
                                          window=1,
                                          step_size=1,
                                          chosen_samples=50,
                                          scaling_factor=adm_parameters['Ripple_sf'][0][0]))

    fr_threshold = np.ceil(find_thresholds(signal=fr_signal,
                                           time=signal_time,
                                           window=1,
                                           step_size=1,
                                           chosen_samples=50,
                                           scaling_factor=adm_parameters['FR_sf'][0][0]))

    # ==================================
    # ADM stage
    # ==================================
    # Convert each signal into a stream of up and DOWN spikes
    r_up, r_dn = signal_to_spike_refractory(interpfact=adm_parameters['interpfact'][0][0],
                                            time=signal_time,
                                            amplitude=r_signal,
                                            thr_up=r_threshold, thr_dn=r_threshold,
                                            refractory_period=adm_parameters['refractory'][0][0])
    ripple = Ripple(up=r_up, down=r_dn)

    fr_up, fr_dn = signal_to_spike_refractory(interpfact=adm_parameters['interpfact'][0][0],
                                              time=signal_time,
                                              amplitude=fr_signal,
                                              thr_up=fr_threshold, thr_dn=fr_threshold,
                                              refractory_period=adm_parameters['refractory'][0][0])
    fast_ripple = FastRipple(up=fr_up, down=fr_dn)

    return FilteredSpikes(ripple=ripple, fast_ripple=fast_ripple)
