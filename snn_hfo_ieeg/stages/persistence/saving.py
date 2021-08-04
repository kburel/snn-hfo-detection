import os
import pickle
from pathlib import Path
from snn_hfo_ieeg.stages.persistence.utility import get_persistence_path


def _create_parent_directory(path):
    parent_directory = Path(path).parent.absolute()
    os.makedirs(parent_directory, exist_ok=True)


def save_hfo_detection(user_facing_hfo_detection, saving_path, metadata):
    filepath = get_persistence_path(saving_path, metadata)
    _create_parent_directory(filepath)
    with open(filepath, "wb") as file:
        pickle.dump(user_facing_hfo_detection, file)
