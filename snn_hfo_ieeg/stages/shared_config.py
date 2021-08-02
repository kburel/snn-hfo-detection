from typing import NamedTuple
from enum import Enum, auto


class MeasurementMode(Enum):
    IEEG = auto()
    ECOG = auto()
    SCALP = auto()


class Configuration(NamedTuple):
    data_path: str
    measurement_mode: MeasurementMode
    hidden_neuron_count: int
    calibration_time: float
