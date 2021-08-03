import os
from pathlib import Path
from typing import NamedTuple
import scipy.io as sio
from snn_hfo_ieeg.user_facing_data import HfoDetectionWithAnalytics


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


def _convert_to_dict(object):
    if _is_namedtuple(object):
        return _convert_to_dict(object._asdict())
    if _is_list(object):
        return [_convert_to_dict(_) for _ in object]
    if _is_dict(object):
        for key, item in object.items():
            object[key] = _convert_to_dict(item)
        return object
    # Needed because of https://stackoverflow.com/questions/38934433/maximum-recursion-depth-exceeded-when-using-scipy-io-savemat
    return str(object)


def _get_file_path(saving_path, metadata) -> str:
    parent_directory = os.path.join(saving_path,
                                    f'P{metadata.patient}',
                                    f'I{metadata.interval}')
    os.makedirs(parent_directory, exist_ok=True)
    filename = f'C{metadata.channel}.mat'

    return os.path.join(parent_directory, filename)


def _create_parent_directory(path):
    parent_directory = Path(path).parent.absolute()
    os.makedirs(parent_directory, exist_ok=True)


def save_hfo_detection(user_facing_hfo_detection, saving_path, metadata):
    filepath = _get_file_path(saving_path, metadata)
    _create_parent_directory(filepath)
    dictionary = _convert_to_dict(user_facing_hfo_detection)
    sio.savemat(filepath, dictionary)


def _from_dict(dictionary: dict, type: NamedTuple) -> HfoDetectionWithAnalytics:
    params = {}
    for key, item in dictionary.items():
        item_type = type.__annotations__[key]
        typed_item = _from_dict(item, item_type) if isinstance(
            item, dict) else item_type(item)
        params[key] = typed_item
    return type(**params)


def load_hfo_detection(loading_path, metadata) -> HfoDetectionWithAnalytics:
    filepath = _get_file_path(loading_path, metadata)
    dictionary = sio.loadmat(filepath)
    return _from_dict(dictionary, HfoDetectionWithAnalytics)
