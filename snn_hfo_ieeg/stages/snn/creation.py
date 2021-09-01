
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.core.groups import Neurons, Connections
from brian2.units import us, amp
from snn_hfo_ieeg.functions.dynapse_biases import get_current


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
        from_layer, to_layer, equation_builder=equation_builder, name=f'{name}synapses', verbose=False, dt=100*us)
    synapses.connect()
    synapses.weight = weights
    synapses.I_tau = get_current(taus*1e-3) * amp
    return synapses
