import os
import json
from pathlib import Path
import numpy as np
from snn_hfo_ieeg.stages.persistence.utility import get_persistence_path


def _create_parent_directory(path):
    parent_directory = Path(path).parent.absolute()
    os.makedirs(parent_directory, exist_ok=True)


def _is_namedtuple(obj) -> bool:
    # Source: https://stackoverflow.com/a/62692640/5903309
    return (
        isinstance(obj, tuple) and
        hasattr(obj, '_asdict') and
        hasattr(obj, '_fields')
    )


def _is_list(obj) -> bool:
    return (
        hasattr(obj, '__iter__') and
        not isinstance(obj, (str, dict))
    )


def _is_dict(obj) -> bool:
    return isinstance(obj, dict)


def _is_np_bool(obj) -> bool:
    return isinstance(obj, np.bool_)


def _convert_to_dict(object):
    if _is_namedtuple(object):
        return _convert_to_dict(object._asdict())
    if _is_list(object):
        return [_convert_to_dict(_) for _ in object]
    if _is_dict(object):
        for key, item in object.items():
            object[key] = _convert_to_dict(item)
        return object
    if _is_np_bool(object):
        return bool(object)
    if object is None:
        return str(object)
    return object


def save_hfo_detection(user_facing_hfo_detection, saving_path, metadata):
    filepath = get_persistence_path(saving_path, metadata)
    _create_parent_directory(filepath)
    dictionary = _convert_to_dict(user_facing_hfo_detection)
    with open(filepath, 'w') as file:
        json.dump(dictionary, file)
