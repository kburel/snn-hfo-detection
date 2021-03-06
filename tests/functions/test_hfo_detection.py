import pytest
from snn_hfo_detection.functions.hfo_detection import Periods, detect_hfo, HfoDetectionWithAnalytics, HfoDetection, Analytics
from tests.utility import *


@pytest.mark.parametrize(
    'duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection',
    [(1, [0], np.array([0]), 0.1, 0.1, HfoDetectionWithAnalytics(
        result=HfoDetection(
            total_amount=1,
            frequency=1,
        ),
        analytics=Analytics(
            detections=[1],
            periods=Periods(
                start=[0],
                stop=[0]
            )
        ))),
     (1, [0.5], np.array([0.5]), 0.1, 0.5, HfoDetectionWithAnalytics(
         result=HfoDetection(
             total_amount=1,
             frequency=1,
         ),
         analytics=Analytics(
             detections=[1],
             periods=Periods(
                 start=[0.5],
                 stop=[0.5]
             )
         ))),
     (2, [-0.5], np.array([-0.5]), -0.1, -0.05, HfoDetectionWithAnalytics(
         result=HfoDetection(
             total_amount=0,
             frequency=0,
         ),
         analytics=Analytics(
             detections=[0],
             periods=Periods(
                 start=[],
                 stop=[]
             )
         ))),
     (0.001, np.arange(0, 300, 5e-4), [0.2, 0.3], 0.01, 0.05, HfoDetectionWithAnalytics(
         result=HfoDetection(
             total_amount=0,
             frequency=0,
         ),
         analytics=Analytics(
             detections=[0, 0],
             periods=Periods(
                 start=[],
                 stop=[]
             )
         )))]
)
def test_hfo_detection(duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection):
    hfo_detection = detect_hfo(duration, spike_monitor,
                               original_time_vector, step_size, window_size)
    assert_are_hfo_detections_equal(expected_hfo_detection, hfo_detection)


def test_hfo_detection_fails_when_step_size_is_bigger_than_window():
    with pytest.raises(ValueError):
        detect_hfo(1, [0], [0], 1, 0.5)


def test_hfo_detection_fails_when_step_size_is_zero():
    with pytest.raises(ZeroDivisionError):
        detect_hfo(1, [0], [0], 0, 0)


def test_hfo_detection_fails_when_duration_is_zero():
    with pytest.raises(ValueError):
        detect_hfo(0, [0], [0], 1, 0.5)
