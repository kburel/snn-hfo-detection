import os
from typing import NamedTuple
import scipy.io as sio
_PACKAGE_NAME = 'snn_hfo_ieeg'


class NetworkParameters(NamedTuple):
    neuron_model_path: str
    synapse_model_path: str
    adm_parameters: dict
    network_parameters: dict


def load_network_parameters():
    parameters_path = os.path.join(_PACKAGE_NAME, 'parameters')
    snn_models_path = os.path.join(_PACKAGE_NAME, 'models')

    # Load SNN parameters, neuron and synapse models
    neuron_model_path = os.path.join(snn_models_path, 'neuron')
    synapse_model_path = os.path.join(snn_models_path, 'synapse')

    adm_parameters = sio.loadmat(os.path.join(
        parameters_path, 'adm.mat'))
    network_parameters = sio.loadmat(
        os.path.join(parameters_path, 'network.mat'))

    return NetworkParameters(
        neuron_model_path=neuron_model_path,
        synapse_model_path=synapse_model_path,
        adm_parameters=adm_parameters,
        network_parameters=network_parameters
    )
