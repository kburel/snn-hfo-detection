import warnings
from brian2 import start_scope, run, SpikeGeneratorGroup, SpikeMonitor
from brian2.units import us, amp, second
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes
from snn_hfo_ieeg.functions.dynapse_biases import get_tau_current
from snn_hfo_ieeg.stages.snn.tau_generation import generate_concatenated_taus
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights
from snn_hfo_ieeg.stages.snn.model_paths import load_model_paths

INPUT_COUNT = 4
HIDDEN_NEURON_COUNT = 86


def _concatenate_filtered_spikes(filtered_spikes):
    spikes_list = {
        'r_up': filtered_spikes.ripple.up,
        'r_dn': filtered_spikes.ripple.down,
        'fr_up': filtered_spikes.fast_ripple.up,
        'fr_dn': filtered_spikes.fast_ripple.down
    }
    return concatenate_spikes(spikes_list)


def _create_input_layer(filtered_spikes):
    input_spiketimes, input_neurons_id = _concatenate_filtered_spikes(
        filtered_spikes)
    return SpikeGeneratorGroup(INPUT_COUNT,
                               input_neurons_id,
                               input_spiketimes*second,
                               dt=100*us, name='input')


def _create_hidden_layer(model_paths):
    hidden_neurons = HIDDEN_NEURON_COUNT
    equation_builder = NeuronEquationBuilder.import_eq(
        model_paths.neuron, num_inputs=1)
    hidden_layer = Neurons(
        hidden_neurons, equation_builder=equation_builder, name='hidden_layer', dt=100*us)
    return hidden_layer


def _create_synapses(input_layer, hidden_layer, model_paths):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        input_layer, hidden_layer, equation_builder=equation_builder, name='synapses', verbose=False, dt=100*us)

    synapses.connect()

    synapses.weight = generate_weights(INPUT_COUNT, HIDDEN_NEURON_COUNT)
    taus = generate_concatenated_taus(INPUT_COUNT, HIDDEN_NEURON_COUNT)
    synapses.I_tau = get_tau_current(taus*1e-3, True) * amp

    return synapses


def snn_stage(filtered_spikes, duration):
    warnings.simplefilter("ignore", DeprecationWarning)
    start_scope()

    model_paths = load_model_paths()
    input_layer = _create_input_layer(filtered_spikes)
    hidden_layer = _create_hidden_layer(model_paths)
    # Do not remove the following variable, otherwise its lifetime is too narrow
    _synapses = _create_synapses(
        input_layer, hidden_layer, model_paths)

    spike_monitor_hidden = SpikeMonitor(hidden_layer)
    run(duration * second)

    return spike_monitor_hidden
