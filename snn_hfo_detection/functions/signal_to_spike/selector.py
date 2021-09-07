from enum import Enum, auto
from snn_hfo_detection.functions.signal_to_spike.utility import SignalToSpikeParameters, SpikeTrains
from snn_hfo_detection.functions.signal_to_spike import default, realistic


class SignalToSpikeAlgorithm(Enum):
    DEFAULT = auto()
    REALISTIC = auto()


def signal_to_spike(parameters: SignalToSpikeParameters, algorithm: SignalToSpikeAlgorithm) -> SpikeTrains:
    if algorithm is SignalToSpikeAlgorithm.DEFAULT:
        return default.signal_to_spike(parameters)
    if algorithm is SignalToSpikeAlgorithm.REALISTIC:
        return realistic.signal_to_spike(parameters)
    raise ValueError(f"Unknown algorithm: {algorithm}")
