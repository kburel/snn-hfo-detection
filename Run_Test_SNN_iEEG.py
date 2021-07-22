from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from SNN_HFO_iEEG.Functions.filter import *
from SNN_HFO_iEEG.Functions.dynapse_biases import *
from SNN_HFO_iEEG.Functions.signal_to_spike import *
from SNN_HFO_iEEG.Functions.hfo_detection import *
import os
import argparse
import scipy.io as sio
from brian2 import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

_PACKAGE_NAME = 'SNN_HFO_iEEG'


def run_hfo_detection(data_path, hfo_callback):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parameters_path = os.path.join(_PACKAGE_NAME, 'Parameters')
    snn_models_path = os.path.join(_PACKAGE_NAME, 'Models')

    # Load SNN parameters, neuron and synapse models
    neuron_model_path = os.path.join(snn_models_path, 'neuron')
    synapse_model_path = os.path.join(snn_models_path, 'synapse')

    ADM_parameters = sio.loadmat(os.path.join(
        parameters_path, 'ADM_parameters.mat'))
    Network_parameters = sio.loadmat(
        os.path.join(parameters_path, 'Network_parameters.mat'))

    sampling_frequency = 2000

    # Select Data from a single patient
    patient = 1
    # Interval
    current_interval = 1

    Interval = sio.loadmat(os.path.join(data_path, 'P%s/P%sI%s' %
                                        (patient, patient, current_interval) + '.mat'))

    num_channels = Interval['chb'].shape[0]
    for ch in range(num_channels):

        print('Running test for Patient %s, interval %s and channel %s' %
              (patient, current_interval, ch))

        # Prepare dictionaries to return results
        if ch == 0:
            Test_info = {}
            Test_info['Patient'] = patient
            Test_info['Interval'] = current_interval
            Test_info['Channels'] = num_channels

            Test_Results = {}
            Test_Results['Info'] = Test_info
            Test_Results['SNN'] = {}
            Test_Results['SNN']['number_HFO'] = np.zeros(num_channels)
            Test_Results['SNN']['rate_HFO'] = np.zeros(num_channels)

        # ==================================
        # Filtering stage
        # ==================================
        # Get the data for the current channel
        Wideband_signal = Interval['chb'][ch]
        signal_time = Interval['t'][0]

        # Prepare dictionaries to return data
        if ch == 0:
            Signal = {}
            Signal['time'] = signal_time
            Signal['Ripple'] = np.zeros((num_channels, signal_time.size))
            Signal['FR'] = np.zeros((num_channels, signal_time.size))

            Spikes = {}
            Spikes['Ripple'] = {}
            Spikes['Ripple']['thresholds'] = np.zeros(num_channels)
            Spikes['FR'] = {}
            Spikes['FR']['thresholds'] = np.zeros(num_channels)

        # Filter the Wideband in Ripple and FR bands
        R_signal = butter_bandpass_filter(data=Wideband_signal,
                                          lowcut=80,
                                          highcut=250,
                                          fs=sampling_frequency,
                                          order=2)
        FR_signal = butter_bandpass_filter(data=Wideband_signal,
                                           lowcut=250,
                                           highcut=500,
                                           fs=sampling_frequency,
                                           order=2)

        Signal['Ripple'][ch, :] = R_signal
        Signal['FR'][ch, :] = FR_signal

        # ==================================
        # Baseline detection stage
        # ==================================
        # Based on the noise floor find the signal-to-spike thresholds
        R_threshold = np.ceil(find_thresholds(signal=R_signal,
                                              time=signal_time,
                                              window=1,
                                              step_size=1,
                                              chosen_samples=50,
                                              scaling_factor=ADM_parameters['Ripple_sf'][0][0]))

        FR_threshold = np.ceil(find_thresholds(signal=FR_signal,
                                               time=signal_time,
                                               window=1,
                                               step_size=1,
                                               chosen_samples=50,
                                               scaling_factor=ADM_parameters['FR_sf'][0][0]))

        Spikes['Ripple']['thresholds'] = R_threshold
        Spikes['FR']['thresholds'] = FR_threshold

        # ==================================
        # ADM stage
        # ==================================
        # Convert each signal into a stream of UP and DOWN spikes
        R_up, R_dn = signal_to_spike_refractory(interpfact=ADM_parameters['interpfact'][0][0],
                                                time=signal_time,
                                                amplitude=R_signal,
                                                thr_up=R_threshold, thr_dn=R_threshold,
                                                refractory_period=ADM_parameters['refractory'][0][0])

        FR_up, FR_dn = signal_to_spike_refractory(interpfact=ADM_parameters['interpfact'][0][0],
                                                  time=signal_time,
                                                  amplitude=FR_signal,
                                                  thr_up=FR_threshold, thr_dn=FR_threshold,
                                                  refractory_period=ADM_parameters['refractory'][0][0])

        Spikes['Ripple']['ch_%s' % ch] = {}
        Spikes['Ripple']['ch_%s' % ch]['UP'] = R_up
        Spikes['Ripple']['ch_%s' % ch]['DN'] = R_dn

        Spikes['FR']['ch_%s' % ch] = {}
        Spikes['FR']['ch_%s' % ch]['UP'] = FR_up
        Spikes['FR']['ch_%s' % ch]['DN'] = FR_dn

        # ==================================
        # SNN stage
        # ==================================
        # Spikes in SNN format
        spikes_list = {}
        spikes_list['R_up'] = R_up
        spikes_list['R_dn'] = R_dn
        spikes_list['FR_up'] = FR_up
        spikes_list['FR_dn'] = FR_dn
        input_spiketimes, input_neurons_ID = concatenate_spikes(spikes_list)

        extra_simulation_time = 0.050
        #-----------% SNN Input %-----------#
        start_scope()

        Input_channels = Network_parameters['input_neurons'][0][0]
        Input = SpikeGeneratorGroup(Input_channels,
                                    input_neurons_ID,
                                    input_spiketimes*second,
                                    dt=100*us, name='Input')

        #-----------% SNN hidden layer neurons %-----------#
        Hidden_neurons = Network_parameters['hidden_neurons'][0][0]
        builder_object1 = NeuronEquationBuilder.import_eq(
            neuron_model_path, num_inputs=1)
        Hidden_layer = Neurons(
            Hidden_neurons, equation_builder=builder_object1, name='Hidden_layer', dt=100*us)
        Hidden_layer.refP = Network_parameters['neuron_refractory'][0][0] * second
        Hidden_layer.Itau = getTauCurrent(
            Network_parameters['neuron_taus'][0][0]*1e-3, False) * amp

        #-----------% SNN Synapse %-----------#
        builder_object2 = SynapseEquationBuilder.import_eq(synapse_model_path)
        Input_Hidden_layer = Connections(
            Input, Hidden_layer, equation_builder=builder_object2, name='Input_Hidden_layer', verbose=False, dt=100*us)

        Input_Hidden_layer.connect()
        Input_Hidden_layer.weight = Network_parameters['synapse_weights'][0]
        Input_Hidden_layer.I_tau = getTauCurrent(
            Network_parameters['synapse_taus'][0]*1e-3, True) * amp
        Input_Hidden_layer.baseweight = 1 * pamp

        #-----------% SNN Monitors %-----------#
        Spike_Monitor_Hidden = SpikeMonitor(Hidden_layer)
        Hidden_Monitor = StateMonitor(Hidden_layer, variables=['Iin'], record=[
            0], name='Test_Hidden_Monitor')

        # Run SNN simulation
        duration = np.max(signal_time) + extra_simulation_time
        duration = 0.001
        print('SNN simulation will run for ', duration, ' seconds')
        run(duration * second)

        # ==================================
        # Readout stage
        # ==================================
        print('Running HFO detection')
        HFO_detection_step_size = 0.01
        HFO_detection_window_size = 0.05
        HFO_detection = detect_HFO(trial_duration=duration,
                                   spike_monitor=(
                                       Spike_Monitor_Hidden.t/second),
                                   original_time_vector=signal_time,
                                   step_size=HFO_detection_step_size,
                                   window_size=HFO_detection_window_size)

        detected_HFO = HFO_detection['total_HFO']

        # Save HFO results
        Test_Results['SNN']['number_HFO'][ch] = detected_HFO
        Test_Results['SNN']['rate_HFO'][ch] = detected_HFO/duration

        print('Found HFO', detected_HFO)
        print('Rate of HFO (event/min)',
              np.around((detected_HFO/duration)*60, decimals=2))
        print(' ')

        hfo_callback(HFO_detection)


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Perform an HFO test run')
    parser.add_argument('--data-path', type=str, nargs='?', default='Data/',
                        help='Specifies the path to the directory containing the test data. Default is ./Data/')
    return parser.parse_args()


if __name__ == '__main__':
    data_path = _parse_arguments().data_path
    run_hfo_detection(data_path, lambda _: {})
