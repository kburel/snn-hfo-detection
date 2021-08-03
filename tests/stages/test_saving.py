from shutil import rmtree
import numpy as np
from snn_hfo_ieeg.user_facing_data import Analytics, FilteredSpikes, HfoDetection, HfoDetectionWithAnalytics, Periods, SpikeTrains
from snn_hfo_ieeg.stages.saving import save_hfo_detection, load_hfo_detection
from snn_hfo_ieeg.entrypoint.hfo_detection import Metadata
from tests.utility import assert_are_hfo_detections_equal

SAVED_HFO_DETECTION = HfoDetectionWithAnalytics(
    result=HfoDetection(
        frequency=1.0,
        total_amount=2.0,
    ),
    analytics=Analytics(
        detections=np.array([1, 2]),
        analyzed_times=np.array([0, 1, 2, 3]),
        periods=Periods(
            start=np.array([1, 2]),
            stop=np.array([1.5, 3])
        ),
        filtered_spikes=FilteredSpikes(
            ripple=None,
            fast_ripple=SpikeTrains(
                up=np.array([1.5, 2.1]),
                down=np.array([2.4, 3])
            )),
        spike_times=np.array([1, 2, 2.3, 3]),
        neuron_ids=np.array([0, 1, 2, 2])
    ))

METADATA = Metadata(
    patient=1,
    interval=2,
    channel=3,
    duration=1.3
)

SAVING_PATH = 'output_from_test_saved_can_be_loaded/'


def _assert_saved_can_be_loaded():
    save_hfo_detection(
        user_facing_hfo_detection=SAVED_HFO_DETECTION,
        saving_path=SAVING_PATH,
        metadata=METADATA
    )
    loaded_hfo_detection = load_hfo_detection(
        loading_path=SAVING_PATH,
        metadata=METADATA
    )
    print(loaded_hfo_detection)

    assert_are_hfo_detections_equal(
        SAVED_HFO_DETECTION, loaded_hfo_detection)


def test_saved_can_be_loaded():
    try:
        _assert_saved_can_be_loaded()
    finally:
        rmtree(SAVING_PATH)
