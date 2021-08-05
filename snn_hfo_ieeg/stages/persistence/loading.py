import json
from os import path
from types import SimpleNamespace
from snn_hfo_ieeg.user_facing_data import HfoDetectionWithAnalytics
from snn_hfo_ieeg.stages.persistence.utility import get_persistence_path


def load_hfo_detection(loading_path, metadata) -> HfoDetectionWithAnalytics:
    filepath = get_persistence_path(loading_path, metadata)
    if not path.isfile(filepath):
        raise ValueError(
            f'No HFO detection data was saved for patient {metadata.patient}, interval {metadata.interval}, channel {metadata.channel}')
    with open(filepath, 'r') as file:
        return json.load(file, object_hook=lambda d: SimpleNamespace(**d))
