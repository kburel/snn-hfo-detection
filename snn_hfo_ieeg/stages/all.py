from brian2.units import second
import numpy as np
from snn_hfo_ieeg.stages.filter import filter_stage
from snn_hfo_ieeg.stages.snn.stage import snn_stage
from snn_hfo_ieeg.functions.hfo_detection import detect_hfo
from snn_hfo_ieeg.user_facing_data import HfoDetection, HfoDetectionWithAnalytics, Analytics
from snn_hfo_ieeg.stages.persistence.saving import save_hfo_detection

HFO_DETECTION_STEP_SIZE = 0.01
HFO_DETECTION_WINDOW_SIZE = 0.05


def _convert_inner_hfo_detection_to_user_facing_one(hfo_detection, filtered_spikes, spike_monitor_hidden):
    return HfoDetectionWithAnalytics(
        result=HfoDetection(
            total_amount=hfo_detection.result.total_amount,
            frequency=hfo_detection.result.frequency,
        ),
        analytics=Analytics(
            detections=hfo_detection.analytics.detections,
            periods=hfo_detection.analytics.periods,
            filtered_spikes=filtered_spikes,
            spike_times=np.array(spike_monitor_hidden.t/second),
            neuron_ids=np.array(spike_monitor_hidden.i),
        )
    )


def run_all_hfo_detection_stages(metadata, channel_data, duration, configuration, snn_cache):
    filtered_spikes = filter_stage(channel_data, configuration)
    spike_monitor_hidden = snn_stage(
        filtered_spikes=filtered_spikes,
        duration=duration,
        configuration=configuration,
        cache=snn_cache)

    hfo_detection = detect_hfo(duration=duration,
                               spike_times=(
                                   spike_monitor_hidden.t/second),
                               signal_times=channel_data.signal_time,
                               step_size=HFO_DETECTION_STEP_SIZE,
                               window_size=HFO_DETECTION_WINDOW_SIZE)
    user_facing_hfo_detection = _convert_inner_hfo_detection_to_user_facing_one(
        hfo_detection, filtered_spikes, spike_monitor_hidden)

    if not configuration.disable_saving:
        save_hfo_detection(user_facing_hfo_detection=user_facing_hfo_detection,
                           saving_path=configuration.saving_path,
                           metadata=metadata)
    return user_facing_hfo_detection
