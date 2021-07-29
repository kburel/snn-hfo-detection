from snn_hfo_ieeg.plotting.plot_kinds import PlotKind
from typing import List, NamedTuple
from enum import Enum, auto


class MeasurementMode(Enum):
    IEEG = auto()
    ECOG = auto()
    SCALP = auto()


class Configuration(NamedTuple):
    data_path: str
    measurement_mode: MeasurementMode
    hidden_neuron_count: int
    plots: List[PlotKind]
