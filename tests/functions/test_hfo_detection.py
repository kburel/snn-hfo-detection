import pytest
from snn_hfo_ieeg.functions.hfo_detection import Periods, detect_hfo, HfoDetection, PlottingData
from tests.utility import *


@pytest.mark.parametrize(
    'duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection',
    [(0, [0], np.array([0]), 0.1, 0.1, HfoDetection(
        total_amount=0,
        frequency=0,
        plotting_data=PlottingData(
            detections=[0],
            analyzed_times=[0],
            periods=Periods(
                start=[],
                stop=[]
            )
        ))),
     (1, [0.5], np.array([0.5]), 0.1, 0.5, HfoDetection(
         total_amount=1,
         frequency=1,
         plotting_data=PlottingData(
             detections=[1],
             analyzed_times=[0.5],
             periods=Periods(
                 start=[0.5],
                 stop=[0.5]
             )
         ))),
     (-1, [-0.5], np.array([-0.5]), -0.1, -0.05, HfoDetection(
         total_amount=0,
         frequency=0,
         plotting_data=PlottingData(
             detections=[0],
             analyzed_times=[-0.5],
             periods=Periods(
                 start=[],
                 stop=[]
             )
         ))),
     (0.001, np.arange(0, 300, 5e-4), [0.2, 0.3], 0.01, 0.05, HfoDetection(
         total_amount=0,
         frequency=0,
         plotting_data=PlottingData(
             detections=[0, 0],
             analyzed_times=[0.2, 0.3],
             periods=Periods(
                 start=[],
                 stop=[]
             )
         )))]
)
def test_hfo_detection(duration, spike_monitor, original_time_vector, step_size, window_size, expected_hfo_detection):
    hfo_detection = detect_hfo(duration, spike_monitor,
                               original_time_vector, step_size, window_size)
    assert are_hfo_detections_equal(expected_hfo_detection, hfo_detection)


def test_hfo_detection_fails_when_step_size_is_bigger_than_window():
    with pytest.raises(AssertionError):
        detect_hfo(0, [0], [0], 1, 0.5)


def test_hfo_detection_fails_when_step_size_is_zero():
    with pytest.raises(ZeroDivisionError):
        detect_hfo(0, [0], [0], 0, 0)
