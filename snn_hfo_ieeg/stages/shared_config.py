from typing import NamedTuple
from enum import Enum


class MeasurementMode(Enum):
    IEEG = 1
    ECOG = 2
    SCALP = 3


class Configuration(NamedTuple):
    data_path: str
    measurement_mode: MeasurementMode
    hidden_neuron_count: int
    duration: float
