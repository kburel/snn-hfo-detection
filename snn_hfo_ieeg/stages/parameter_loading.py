import os
from typing import NamedTuple
import scipy.io as sio
_PACKAGE_NAME = 'snn_hfo_ieeg'


class NetworkParameters(NamedTuple):
    '''
    Pre-defined network parameters

    Parameters
    --------
    neuron_model_path : str
        Path to the neuron model imported by teili
    synapse_model_path : str
        Path to the synapse model imported by teili
    adm_parameters : dict
        Imported ADM parameters from .mat file
    imported_network_parameters : dict
        Imported network parameters from .mat file
    '''
    neuron_model_path: str
    synapse_model_path: str
    adm_parameters: dict
    imported_network_parameters: dict


def load_network_parameters():
    parameters_path = os.path.join(_PACKAGE_NAME, 'parameters')
    snn_models_path = os.path.join(_PACKAGE_NAME, 'models')

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
        imported_network_parameters=network_parameters
    )
