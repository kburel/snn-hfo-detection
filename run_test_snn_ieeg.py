from typing import NamedTuple
import os
import warnings
import argparse
import scipy.io as sio
from brian2.units import second
from snn_hfo_ieeg.functions.filter import *
from snn_hfo_ieeg.functions.dynapse_biases import *
from snn_hfo_ieeg.functions.signal_to_spike import *
from snn_hfo_ieeg.functions.hfo_detection import *
from snn_hfo_ieeg.stages.filter import filter_stage
from snn_hfo_ieeg.stages.snn import snn_stage

_PACKAGE_NAME = 'snn_hfo_ieeg'

_SAMPLING_FREQUENCY = 2000


def run_hfo_detection(data_path, hfo_cb):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parameters_path = os.path.join(_PACKAGE_NAME, 'parameters')
    snn_models_path = os.path.join(_PACKAGE_NAME, 'models')

    # Load SNN parameters, neuron and synapse models
    neuron_model_path = os.path.join(snn_models_path, 'neuron')
    synapse_model_path = os.path.join(snn_models_path, 'synapse')

    adm_parameters = sio.loadmat(os.path.join(
        parameters_path, 'adm.mat'))
    network_parameters = sio.loadmat(
        os.path.join(parameters_path, 'network.mat'))

    # Select Data from a single patient
    patient = 1
    # interval
    current_interval = 1

    file_name = f'P{patient}/P{patient}I{current_interval}.mat'
    interval = sio.loadmat(os.path.join(data_path, file_name))

    num_channels = interval['chb'].shape[0]
    for channel in range(num_channels):
        print(
            f'Running test for Patient {patient}, interval {current_interval} and channel {channel}')

        wideband_signal = interval['chb'][channel]
        signal_time = interval['t'][0]

        filtered_spikes = filter_stage(
            wideband_signal=wideband_signal,
            sampling_frequency=_SAMPLING_FREQUENCY,
            signal_time=signal_time,
            adm_parameters=adm_parameters)

        extra_simulation_time = 0.050
        duration = np.max(signal_time) + extra_simulation_time

        spike_monitor_hidden = snn_stage(filtered_spikes=filtered_spikes,
                                         network_parameters=network_parameters,
                                         neuron_model_path=neuron_model_path,
                                         synapse_model_path=synapse_model_path,
                                         duration=duration)
        # ==================================
        # Readout stage
        # ==================================
        print('Running hfo detection')
        hfo_detection_step_size = 0.01
        hfo_detection_window_size = 0.05
        hfo_detection = detect_hfo(trial_duration=duration,
                                   spike_monitor=(
                                       spike_monitor_hidden.t/second),
                                   original_time_vector=signal_time,
                                   step_size=hfo_detection_step_size,
                                   window_size=hfo_detection_window_size)

        detected_hfo = hfo_detection['total_hfo']

        print('Found hfo', detected_hfo)
        print('Rate of hfo (event/min)',
              np.around((detected_hfo/duration)*60, decimals=2))
        print(' ')

        hfo_cb(hfo_detection)


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Perform an hfo test run')
    parser.add_argument('--data-path', type=str, nargs='?', default='data/',
                        help='Specifies the path to the directory containing the test data. Default is ./data/')
    return parser.parse_args()


if __name__ == '__main__':
    data_path = _parse_arguments().data_path
    run_hfo_detection(data_path, lambda _: {})
