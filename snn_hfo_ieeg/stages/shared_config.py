from typing import NamedTuple, Optional
from enum import Enum, auto
from snn_hfo_ieeg.stages.plotting.plot_loader import PlottingFunctions


class MeasurementMode(Enum):
    IEEG = auto()
    ECOG = auto()
    SCALP = auto()


class Configuration(NamedTuple):
    data_path: str
    measurement_mode: MeasurementMode
    hidden_neuron_count: int
    calibration_time: float
    plots: PlottingFunctions
    saving_path: Optional[str]
    disable_saving: bool
    loading_path: Optional[str]
