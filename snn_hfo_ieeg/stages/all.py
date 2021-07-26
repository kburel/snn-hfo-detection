from brian2.units import second
from snn_hfo_ieeg.stages.filter import filter_stage
from snn_hfo_ieeg.stages.snn.stage import snn_stage
from snn_hfo_ieeg.functions.hfo_detection import detect_hfo

HFO_DETECTION_STEP_SIZE = 0.01
HFO_DETECTION_WINDOW_SIZE = 0.05


def run_hfo_detection(channel_data, configuration):
    filtered_spikes = filter_stage(channel_data)

    spike_monitor_hidden = snn_stage(
        filtered_spikes=filtered_spikes,
        configuration=configuration)

    return detect_hfo(trial_duration=configuration.duration,
                      spike_monitor=(
                          spike_monitor_hidden.t/second),
                      original_time_vector=channel_data.signal_time,
                      step_size=HFO_DETECTION_STEP_SIZE,
                      window_size=HFO_DETECTION_WINDOW_SIZE)
