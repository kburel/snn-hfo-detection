from brian2.units import second
from snn_hfo_ieeg.stages.filter import filter_stage
from snn_hfo_ieeg.stages.snn.stage import snn_stage
from snn_hfo_ieeg.functions.hfo_detection import detect_hfo

HFO_DETECTION_STEP_SIZE = 0.01
HFO_DETECTION_WINDOW_SIZE = 0.05


def run_all_hfo_detection_stages(channel_data, duration, configuration, snn_cache):
    filtered_spikes = filter_stage(channel_data, configuration)
    spike_monitor_hidden = snn_stage(
        filtered_spikes=filtered_spikes,
        duration=duration,
        configuration=configuration,
        cache=snn_cache)

    return detect_hfo(duration=duration,
                      spike_times=(
                          spike_monitor_hidden.t/second),
                      signal_times=channel_data.signal_time,
                      step_size=HFO_DETECTION_STEP_SIZE,
                      window_size=HFO_DETECTION_WINDOW_SIZE)
