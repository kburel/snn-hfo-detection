from brian2.units import second
from snn_hfo_ieeg.stages.filter import filter_stage
from snn_hfo_ieeg.stages.snn import snn_stage
from snn_hfo_ieeg.functions.hfo_detection import detect_hfo

SAMPLING_FREQUENCY = 2000
HFO_DETECTION_STEP_SIZE = 0.01
HFO_DETECTION_WINDOW_SIZE = 0.05

def run_hfo_detection(wideband_signal, signal_time, duration, network_parameters):
    filtered_spikes = filter_stage(
        wideband_signal=wideband_signal,
        sampling_frequency=SAMPLING_FREQUENCY,
        signal_time=signal_time,
        adm_parameters=network_parameters.adm_parameters)

    spike_monitor_hidden = snn_stage(filtered_spikes=filtered_spikes,
                                     network_parameters=network_parameters,
                                     neuron_model_path=network_parameters.neuron_model_path,
                                     synapse_model_path=network_parameters.synapse_model_path,
                                     duration=duration)

    return detect_hfo(trial_duration=duration,
                      spike_monitor=(
                          spike_monitor_hidden.t/second),
                      original_time_vector=signal_time,
                      step_size=HFO_DETECTION_STEP_SIZE,
                      window_size=HFO_DETECTION_WINDOW_SIZE)
