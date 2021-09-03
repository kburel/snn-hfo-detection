import numpy as np
from brian2 import SpikeGeneratorGroup
from brian2.units import us, amp, second
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.core.groups import Neurons, Connections
from snn_hfo_ieeg.stages.snn.tau_generation import generate_concatenated_taus
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights
from snn_hfo_ieeg.functions.dynapse_biases import get_current


def create_input_layer(input_count):
    return SpikeGeneratorGroup(N=input_count,
                               indices=[0],
                               times=[0] * second,
                               dt=100*us,
                               name='input_layer')


def create_input_to_hidden_synapses(name, input_layer, hidden_layer, model_paths, neuron_counts):
    weights = generate_weights(neuron_counts)
    taus = generate_concatenated_taus(neuron_counts)
    return create_synapses(
        f'{name}_to_hidden', model_paths, input_layer, hidden_layer, weights, taus)


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
