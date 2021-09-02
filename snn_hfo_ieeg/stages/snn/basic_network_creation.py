from functools import reduce
import numpy as np
from brian2 import SpikeGeneratorGroup
from brian2.units import us, second, amp
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.core.groups import Neurons, Connections
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes
from snn_hfo_ieeg.stages.snn.tau_generation import generate_concatenated_taus
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights
from snn_hfo_ieeg.functions.dynapse_biases import get_current


def _append_spikes(spikes, spike_train):
    spikes.append(spike_train.up)
    spikes.append(spike_train.down)
    return spikes


def _concatenate_bandwidths(bandwidths):
    spike_trains = [
        bandwidth.spike_trains for bandwidth in bandwidths if bandwidth is not None]
    spikes = reduce(_append_spikes, spike_trains, [])
    return concatenate_spikes(spikes)


def create_input_layer(name, bandwidths, input_count):
    input_spiketimes, input_neurons_id = _concatenate_bandwidths(
        bandwidths)

    return SpikeGeneratorGroup(input_count,
                               input_neurons_id,
                               input_spiketimes*second,
                               dt=100*us,
                               name=f'{name}_input_layer')


def create_input_to_hidden_synapses(name, input_layer, hidden_layer, cache):
    weights = generate_weights(cache.neuron_counts)
    taus = generate_concatenated_taus(cache.neuron_counts)
    return create_synapses(
        f'{name}_to_hidden', cache.model_paths, input_layer, hidden_layer, weights, taus)


def create_hidden_to_output_synapses(name, hidden_layer, output_layer, model_paths, hidden_neuron_count):
    weights = np.repeat(3_000.0, hidden_neuron_count.hidden)
    taus = np.repeat(10, hidden_neuron_count.hidden)
    return create_synapses(
        f'{name}_hidden_to_{name}_output', model_paths, hidden_layer, output_layer, weights, taus)


def create_non_input_layer(model_paths, neuron_count, name, num_inputs=1):
    equation_builder = NeuronEquationBuilder.import_eq(
        model_paths.neuron, num_inputs)
    return Neurons(
        N=neuron_count,
        equation_builder=equation_builder,
        name=f'{name}_layer',
        dt=100*us)


def create_synapses(name, model_paths, from_layer, to_layer, weights, taus):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        from_layer, to_layer, equation_builder=equation_builder, name=f'{name}_synapses', verbose=False, dt=100*us)
    synapses.connect()
    synapses.weight = weights
    synapses.I_tau = get_current(taus*1e-3) * amp
    return synapses
