import os
from typing import NamedTuple
_PACKAGE_NAME = 'snn_hfo_detection'


class ModelPaths(NamedTuple):
    '''
    Paths to the SNN models

    Parameters
    --------
    neuron : str
        Path to the neuron model imported by teili
    synapse : str
        Path to the synapse model imported by teili
    '''
    neuron: str
    synapse: str


def load_model_paths():
    snn_models_path = os.path.join(_PACKAGE_NAME, 'models')

    neuron = os.path.join(snn_models_path, 'neuron')
    synapse = os.path.join(snn_models_path, 'synapse')

    return ModelPaths(
        neuron=neuron,
        synapse=synapse
    )
