from brian2.units import second
from snn_hfo_ieeg.stages.filter import filter_stage
from snn_hfo_ieeg.stages.snn.stage import snn_stage
from snn_hfo_ieeg.functions.hfo_detection import detect_hfo
from snn_hfo_ieeg.user_facing_data import HfoDetection, HfoDetectionWithAnalytics, Analytics

HFO_DETECTION_STEP_SIZE = 0.01
HFO_DETECTION_WINDOW_SIZE = 0.05


def _convert_inner_hfo_detection_to_user_facing_one(hfo_detection, filtered_spikes):
    return HfoDetectionWithAnalytics(
        result=HfoDetection(
            total_amount=hfo_detection.result.total_amount,
            frequency=hfo_detection.result.frequency,
        ),
        analytics=Analytics(
            detections=hfo_detection.analytics.detections,
            analyzed_times=hfo_detection.analytics.analyzed_times,
            periods=hfo_detection.analytics.periods,
            filtered_spikes=filtered_spikes
        )
    )


def run_all_hfo_detection_stages(channel_data, duration, configuration):
    filtered_spikes = filter_stage(channel_data, configuration)
    spike_monitor_hidden = snn_stage(
        filtered_spikes=filtered_spikes,
        duration=duration,
        configuration=configuration)

    hfo_detection = detect_hfo(duration=duration,
                               spike_times=(
                                   spike_monitor_hidden.t/second),
                               signal_times=channel_data.signal_time,
                               step_size=HFO_DETECTION_STEP_SIZE,
                               window_size=HFO_DETECTION_WINDOW_SIZE)
    return _convert_inner_hfo_detection_to_user_facing_one(hfo_detection, filtered_spikes)
