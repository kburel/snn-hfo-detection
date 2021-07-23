from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from brian2 import start_scope, run, SpikeGeneratorGroup, SpikeMonitor
from brian2.units import us, amp, pamp, second
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes
from snn_hfo_ieeg.functions.dynapse_biases import get_tau_current


def _concatenate_filtered_spikes(filtered_spikes):
    spikes_list = {
        'r_up': filtered_spikes.ripple.up,
        'r_dn': filtered_spikes.ripple.down,
        'fr_up': filtered_spikes.fast_ripple.up,
        'fr_dn': filtered_spikes.fast_ripple.down
    }
    return concatenate_spikes(spikes_list)


def _create_input_layer(filtered_spikes, network_parameters):
    input_spiketimes, input_neurons_id = _concatenate_filtered_spikes(
        filtered_spikes)
    input_channels = network_parameters['input_neurons'][0][0]
    return SpikeGeneratorGroup(input_channels,
                               input_neurons_id,
                               input_spiketimes*second,
                               dt=100*us, name='input')


def _create_hidden_layer(network_parameters):
    hidden_neurons = network_parameters['hidden_neurons'][0][0]
    builder_object1 = NeuronEquationBuilder.import_eq(
        network_parameters.neuron_model_path, num_inputs=1)
    hidden_layer = Neurons(
        hidden_neurons, equation_builder=builder_object1, name='hidden_layer', dt=100*us)
    hidden_layer.refP = network_parameters['neuron_refractory'][0][0] * second
    hidden_layer.Itau = get_tau_current(
        network_parameters['neuron_taus'][0][0]*1e-3, False) * amp
    return hidden_layer


def _create_input_hidden_layer(input_layer, hidden_layer, network_parameters):
    #-----------% SNN Synapse %-----------#
    builder_object2 = SynapseEquationBuilder.import_eq(
        network_parameters.synapse_model_path)
    input_hidden_layer = Connections(
        input_layer, hidden_layer, equation_builder=builder_object2, name='input_hidden_layer', verbose=False, dt=100*us)

    input_hidden_layer.connect()
    input_hidden_layer.weight = network_parameters['synapse_weights'][0]
    input_hidden_layer.I_tau = get_tau_current(
        network_parameters['synapse_taus'][0]*1e-3, True) * amp
    input_hidden_layer.baseweight = 1 * pamp

    return input_hidden_layer


def snn_stage(filtered_spikes, network_parameters, duration):
    # spikes in SNN format
    #-----------% SNN input %-----------#
    start_scope()

    input_layer = _create_input_layer(filtered_spikes, network_parameters)
    hidden_layer = _create_hidden_layer(network_parameters)
    _create_input_hidden_layer(
        input_layer, hidden_layer, network_parameters)

    #-----------% SNN Monitors %-----------#
    spike_monitor_hidden = SpikeMonitor(hidden_layer)

    run(duration * second)
    return spike_monitor_hidden