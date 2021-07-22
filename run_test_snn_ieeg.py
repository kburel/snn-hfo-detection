from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from snn_hfo_ieeg.functions.filter import *
from snn_hfo_ieeg.functions.dynapse_biases import *
from snn_hfo_ieeg.functions.signal_to_spike import *
from snn_hfo_ieeg.functions.hfo_detection import *
import os
import argparse
import scipy.io as sio
from brian2 import *
import warnings

_PACKAGE_NAME = 'snn_hfo_ieeg'


def run_hfo_detection(data_path, hfo_callback):
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

    sampling_frequency = 2000

    # Select Data from a single patient
    patient = 1
    # interval
    current_interval = 1

    file_name = f'P{patient}/P{patient}I{current_interval}.mat'
    interval = sio.loadmat(os.path.join(data_path, file_name))

    num_channels = interval['chb'].shape[0]
    for ch in range(num_channels):

        print(
            f'Running test for Patient {patient}, interval {current_interval} and channel {ch}')

        # Prepare dictionaries to return results
        if ch == 0:
            test_info = {}
            test_info['Patient'] = patient
            test_info['interval'] = current_interval
            test_info['Channels'] = num_channels

            test_results = {}
            test_results['Info'] = test_info
            test_results['SNN'] = {}
            test_results['SNN']['number_hfo'] = np.zeros(num_channels)
            test_results['SNN']['rate_hfo'] = np.zeros(num_channels)

        # ==================================
        # Filtering stage
        # ==================================
        # Get the data for the current channel
        wideband_signal = interval['chb'][ch]
        signal_time = interval['t'][0]

        # Prepare dictionaries to return data
        if ch == 0:
            signal = {}
            signal['time'] = signal_time
            signal['ripple'] = np.zeros((num_channels, signal_time.size))
            signal['fr'] = np.zeros((num_channels, signal_time.size))

            spikes = {}
            spikes['ripple'] = {}
            spikes['ripple']['thresholds'] = np.zeros(num_channels)
            spikes['fr'] = {}
            spikes['fr']['thresholds'] = np.zeros(num_channels)

        # Filter the wideband in ripple and fr bands
        r_signal = butter_bandpass_filter(data=wideband_signal,
                                          lowcut=80,
                                          highcut=250,
                                          fs=sampling_frequency,
                                          order=2)
        fr_signal = butter_bandpass_filter(data=wideband_signal,
                                           lowcut=250,
                                           highcut=500,
                                           fs=sampling_frequency,
                                           order=2)

        signal['ripple'][ch, :] = r_signal
        signal['fr'][ch, :] = fr_signal

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

        spikes['ripple']['thresholds'] = r_threshold
        spikes['fr']['thresholds'] = fr_threshold

        # ==================================
        # ADM stage
        # ==================================
        # Convert each signal into a stream of up and DOWN spikes
        r_up, r_dn = signal_to_spike_refractory(interpfact=adm_parameters['interpfact'][0][0],
                                                time=signal_time,
                                                amplitude=r_signal,
                                                thr_up=r_threshold, thr_dn=r_threshold,
                                                refractory_period=adm_parameters['refractory'][0][0])

        fr_up, fr_dn = signal_to_spike_refractory(interpfact=adm_parameters['interpfact'][0][0],
                                                  time=signal_time,
                                                  amplitude=fr_signal,
                                                  thr_up=fr_threshold, thr_dn=fr_threshold,
                                                  refractory_period=adm_parameters['refractory'][0][0])
        channel_key = f'ch_{ch}'
        spikes['ripple'][channel_key] = {
            'up': r_up,
            'dn': r_dn
        }
        spikes['fr'][channel_key] = {
            'up': fr_up,
            'dn': fr_dn
        }

        # ==================================
        # SNN stage
        # ==================================
        # spikes in SNN format
        spikes_list = {}
        spikes_list['r_up'] = r_up
        spikes_list['r_dn'] = r_dn
        spikes_list['fr_up'] = fr_up
        spikes_list['fr_dn'] = fr_dn
        input_spiketimes, input_neurons_ID = concatenate_spikes(spikes_list)

        extra_simulation_time = 0.050
        #-----------% SNN input %-----------#
        start_scope()

        input_channels = network_parameters['input_neurons'][0][0]
        input = SpikeGeneratorGroup(input_channels,
                                    input_neurons_ID,
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
        spike_Monitor_hidden = SpikeMonitor(hidden_layer)
        hidden_Monitor = StateMonitor(hidden_layer, variables=['Iin'], record=[
            0], name='test_hidden_Monitor')

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
                                       spike_Monitor_hidden.t/second),
                                   original_time_vector=signal_time,
                                   step_size=hfo_detection_step_size,
                                   window_size=hfo_detection_window_size)

        detected_hfo = hfo_detection['total_hfo']

        # Save HFO results
        test_results['SNN']['number_hfo'][ch] = detected_hfo
        test_results['SNN']['rate_hfo'][ch] = detected_hfo/duration

        print('Found hfo', detected_hfo)
        print('Rate of hfo (event/min)',
              np.around((detected_hfo/duration)*60, decimals=2))
        print(' ')

        hfo_callback(hfo_detection)


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Perform an hfo test run')
    parser.add_argument('--data-path', type=str, nargs='?', default='data/',
                        help='Specifies the path to the directory containing the test data. Default is ./data/')
    return parser.parse_args()


if __name__ == '__main__':
    data_path = _parse_arguments().data_path
    run_hfo_detection(data_path, lambda _: {})
