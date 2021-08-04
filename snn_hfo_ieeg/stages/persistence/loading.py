from typing import NamedTuple
from os import path
import scipy.io as sio
from snn_hfo_ieeg.user_facing_data import HfoDetectionWithAnalytics
from snn_hfo_ieeg.stages.persistence.utility import get_persistence_path


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
        if len(value) == 0:
            params[property_name] = []
        elif str(value[0]) == 'None':
            params[property_name] = None
        elif _has_annotations(property_type):
            params[property_name] = _deserialize_inner_matlab(
                value, property_type)
        elif property_type is int:
            params[property_name] = int(float(value[0]))
        else:
            params[property_name] = property_type(value[0])
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
    filepath = get_persistence_path(loading_path, metadata)
    if not path.isfile(filepath):
        raise ValueError(
            f'No HFO detection data was saved for interval {metadata.interval}, channel {metadata.channel}')
    dictionary = sio.loadmat(filepath)
    _remove_matlab_keys(dictionary)
    return _deserialize_matlab(dictionary, HfoDetectionWithAnalytics)
