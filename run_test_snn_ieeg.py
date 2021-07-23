from typing import NamedTuple
import os
import warnings
import argparse
import scipy.io as sio
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from brian2 import start_scope, run, SpikeGeneratorGroup, SpikeMonitor
from brian2.units import second, us, amp, pamp
from snn_hfo_ieeg.functions.filter import *
from snn_hfo_ieeg.functions.dynapse_biases import *
from snn_hfo_ieeg.functions.signal_to_spike import *
from snn_hfo_ieeg.functions.hfo_detection import *

_PACKAGE_NAME = 'snn_hfo_ieeg'

_SAMPLING_FREQUENCY = 2000


class Ripple(NamedTuple):
    up: np.array
    down: np.array


class FastRipple(NamedTuple):
    up: np.array
    down: np.array


class FilteredSpikes(NamedTuple):
    ripple: Ripple
    fast_ripple: FastRipple


def filter_stage(wideband_signal, sampling_frequency, signal_time, adm_parameters):
    # Filter the Wideband in ripple and fr bands
    r_signal = butter_bandpass_filter(data=wideband_signal,
                                      lowcut=80,
                                      highcut=250,
                                      sampling_frequency=sampling_frequency,
                                      order=2)
    fr_signal = butter_bandpass_filter(data=wideband_signal,
                                       lowcut=250,
                                       highcut=500,
                                       sampling_frequency=sampling_frequency,
                                       order=2)

    # ==================================
    # Baseline detection stage
    # ==================================
    # Based on the noise floor find the signal-to-spike thresholds
    r_threshold = np.ceil(find_thresholds(signal=r_signal,
                                          time=signal_time,
                                          window=1,
                                          step_size=1,
                                          chosen_samples=50,
                                          scaling_factor=adm_parameters['Ripple_sf'][0][0]))

    fr_threshold = np.ceil(find_thresholds(signal=fr_signal,
                                           time=signal_time,
                                           window=1,
                                           step_size=1,
                                           chosen_samples=50,
                                           scaling_factor=adm_parameters['FR_sf'][0][0]))

    # ==================================
    # ADM stage
    # ==================================
    # Convert each signal into a stream of up and DOWN spikes
    r_up, r_dn = signal_to_spike_refractory(interpfact=adm_parameters['interpfact'][0][0],
                                            time=signal_time,
                                            amplitude=r_signal,
                                            thr_up=r_threshold, thr_dn=r_threshold,
                                            refractory_period=adm_parameters['refractory'][0][0])
    ripple = Ripple(up=r_up, down=r_dn)

    fr_up, fr_dn = signal_to_spike_refractory(interpfact=adm_parameters['interpfact'][0][0],
                                              time=signal_time,
                                              amplitude=fr_signal,
                                              thr_up=fr_threshold, thr_dn=fr_threshold,
                                              refractory_period=adm_parameters['refractory'][0][0])
    fast_ripple = FastRipple(up=fr_up, down=fr_dn)

    return FilteredSpikes(ripple=ripple, fast_ripple=fast_ripple)


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
            wideband_signal, _SAMPLING_FREQUENCY, signal_time, adm_parameters)

        # ==================================
        # SNN stage
        # ==================================
        # spikes in SNN format
        spikes_list = {}
        spikes_list['r_up'] = filtered_spikes.ripple.up
        spikes_list['r_dn'] = filtered_spikes.ripple.down
        spikes_list['fr_up'] = filtered_spikes.fast_ripple.up
        spikes_list['fr_dn'] = filtered_spikes.fast_ripple.down
        input_spiketimes, input_neurons_id = concatenate_spikes(spikes_list)

        extra_simulation_time = 0.050
        #-----------% SNN input %-----------#
        start_scope()

        input_channels = network_parameters['input_neurons'][0][0]
        input = SpikeGeneratorGroup(input_channels,
                                    input_neurons_id,
                                    input_spiketimes*second,
                                    dt=100*us, name='input')

        #-----------% SNN hidden layer neurons %-----------#
        hidden_neurons = network_parameters['hidden_neurons'][0][0]
        builder_object1 = NeuronEquationBuilder.import_eq(
            neuron_model_path, num_inputs=1)
        hidden_layer = Neurons(
            hidden_neurons, equation_builder=builder_object1, name='hidden_layer', dt=100*us)
        hidden_layer.refP = network_parameters['neuron_refractory'][0][0] * second
        hidden_layer.Itau = get_tau_current(
            network_parameters['neuron_taus'][0][0]*1e-3, False) * amp

        #-----------% SNN Synapse %-----------#
        builder_object2 = SynapseEquationBuilder.import_eq(synapse_model_path)
        input_hidden_layer = Connections(
            input, hidden_layer, equation_builder=builder_object2, name='input_hidden_layer', verbose=False, dt=100*us)

        input_hidden_layer.connect()
        input_hidden_layer.weight = network_parameters['synapse_weights'][0]
        input_hidden_layer.I_tau = get_tau_current(
            network_parameters['synapse_taus'][0]*1e-3, True) * amp
        input_hidden_layer.baseweight = 1 * pamp

        #-----------% SNN Monitors %-----------#
        spike_monitor_hidden = SpikeMonitor(hidden_layer)

        # Run SNN simulation
        duration = np.max(signal_time) + extra_simulation_time
        # duration = 0.001
        print('SNN simulation will run for ', duration, ' seconds')
        run(duration * second)

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
