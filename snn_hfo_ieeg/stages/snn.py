from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from brian2 import start_scope, run, SpikeGeneratorGroup, SpikeMonitor
from brian2.units import us, amp, pamp, second
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes
from snn_hfo_ieeg.functions.dynapse_biases import get_tau_current


def snn_stage(filtered_spikes, network_parameters, neuron_model_path, synapse_model_path, duration):
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
    # duration = 0.001
    print('SNN simulation will run for ', duration, ' seconds')
    run(duration * second)
    return spike_monitor_hidden
