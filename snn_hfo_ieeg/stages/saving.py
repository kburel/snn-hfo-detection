import os
from pathlib import Path
from typing import NamedTuple
import numpy as np
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


def _has_annotations(type) -> bool:
    return hasattr(type, '__annotations__')


def _is_optional(type) -> bool:
    return (
        hasattr(type, "__args__")
        and len(type.__args__) == 2
    )


def _deserialize_inner_matlab(item, type: NamedTuple):
    values = item[0][0]
    names = item.dtype.names
    params = dict(zip(names, values))
    for property_name, outer_property_type in type.__annotations__.items():
        value = params[property_name]
        property_type = outer_property_type.__args__[0] if _is_optional(
            outer_property_type) else outer_property_type
        if np.all(value == ['None']):
            params[property_name] = None
        elif _has_annotations(property_type):
            params[property_name] = _deserialize_inner_matlab(
                value, property_type)
        elif property_type is int:
            params[property_name] = int(float(value))
        else:
            params[property_name] = property_type(value)
    return type(**params)


def _deserialize_matlab(dictionary: dict, type: NamedTuple) -> HfoDetectionWithAnalytics:
    params = {}
    for key, item in dictionary.items():
        item_type = type.__annotations__[key]
        params[key] = _deserialize_inner_matlab(item, item_type)
    return type(**params)


def _remove_matlab_keys(dictionary):
    for key in ['__header__', '__version__', '__globals__']:
        del dictionary[key]


def load_hfo_detection(loading_path, metadata) -> HfoDetectionWithAnalytics:
    filepath = _get_file_path(loading_path, metadata)
    dictionary = sio.loadmat(filepath)
    _remove_matlab_keys(dictionary)
    return _deserialize_matlab(dictionary, HfoDetectionWithAnalytics)
