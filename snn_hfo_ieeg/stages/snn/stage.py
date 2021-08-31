from functools import reduce
from typing import NamedTuple, Optional
import warnings
from brian2 import Network, SpikeGeneratorGroup, SpikeMonitor
from brian2.input.poissongroup import PoissonGroup
from brian2.units import us, amp, second, Hz
import numpy as np
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes
from snn_hfo_ieeg.functions.dynapse_biases import get_current
from snn_hfo_ieeg.stages.snn.tau_generation import generate_concatenated_taus
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights
from snn_hfo_ieeg.stages.snn.model_paths import ModelPaths, load_model_paths
from snn_hfo_ieeg.stages.snn.concatenation import NeuronCount
from snn_hfo_ieeg.user_facing_data import MeasurementMode


class SpikeMonitors(NamedTuple):
    hidden: SpikeMonitor
    output: SpikeMonitor


def _append_spikes(spikes, spike_train):
    spikes.append(spike_train.up)
    spikes.append(spike_train.down)
    return spikes


def _concatenate_filtered_spikes(filtered_spikes):
    spike_trains = [
        bandwidth.spike_trains for bandwidth in filtered_spikes if bandwidth is not None]
    spikes = reduce(_append_spikes, spike_trains, [])
    return concatenate_spikes(spikes)


def _measurement_mode_to_input_count(measurement_mode):
    if measurement_mode is MeasurementMode.IEEG:
        return 4
    if measurement_mode is MeasurementMode.ECOG:
        return 2
    if measurement_mode is MeasurementMode.SCALP:
        return 2
    raise ValueError(
        f'measurement_mode is outside valid range. Allowed values: {MeasurementMode}, instead got: {measurement_mode}')


def _read_neuron_counts(configuration):
    input_count = _measurement_mode_to_input_count(
        configuration.measurement_mode)
    return NeuronCount(input_count, configuration.hidden_neuron_count)


def _create_input_layer(filtered_spikes, input_count):
    input_spiketimes, input_neurons_id = _concatenate_filtered_spikes(
        filtered_spikes)
    return SpikeGeneratorGroup(input_count,
                               input_neurons_id,
                               input_spiketimes*second,
                               dt=100*us, name='input')


def _create_non_input_layer(model_paths, neuron_count, name, num_inputs=1):
    equation_builder = NeuronEquationBuilder.import_eq(
        model_paths.neuron, num_inputs)
    return Neurons(
        N=neuron_count,
        equation_builder=equation_builder,
        name=f'{name}_layer',
        dt=100*us)


def _create_input_to_hidden_synapses(input_layer, hidden_layer, model_paths, neuron_counts):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        input_layer, hidden_layer, equation_builder=equation_builder, name='input_to_hidden_synapses', verbose=False, dt=100*us)

    synapses.connect()

    synapses.weight = generate_weights(neuron_counts)
    taus = generate_concatenated_taus(neuron_counts)
    synapses.I_tau = get_current(taus*1e-3) * amp

    return synapses


def _create_hidden_to_output_synapses(hidden_layer, output_layer, model_paths, hidden_neuron_count):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        hidden_layer, output_layer, equation_builder=equation_builder, name='hidden_to_output_synapses', verbose=False, dt=100*us)
    synapses.connect()

    synapses.weight = np.repeat(1_000_000.0, hidden_neuron_count.hidden)
    taus = np.repeat(0.01, hidden_neuron_count.hidden)
    synapses.I_tau = get_current(taus*1e-3) * amp
    return synapses


def _create_inhibitor_generator():
    return PoissonGroup(1, 135*Hz, name='inhibitor_generator', dt=100*us)


def _create_input_to_interneuron_synapses(input_layer, interneuron_layer, model_paths):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        input_layer, interneuron_layer, equation_builder=equation_builder, name='input_to_interneuron_synapses', verbose=False, dt=100*us)
    synapses.connect()
    synapses.weight = 3_000
    taus = 5
    synapses.I_tau = get_current(taus*1e-3) * amp
    return synapses


def _create_interneuron_to_inhibitor_synapses(interneuron_layer, inhibitor_layer, model_paths):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        interneuron_layer, inhibitor_layer, equation_builder=equation_builder, name='interneuron_to_inhibitor_synapses', verbose=False, dt=100*us)
    synapses.connect()
    synapses.weight = -25_000
    taus = 20
    synapses.I_tau = get_current(taus*1e-3) * amp
    return synapses


def _create_inhibitor_generator_to_inhibitor_synapses(inhibitor_generator, inhibitor_layer, model_paths):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        inhibitor_generator, inhibitor_layer, equation_builder=equation_builder, name='inhibitor_generator_to_inhibitor_synapses', verbose=False, dt=100*us)
    synapses.connect()
    synapses.weight = 50_000
    taus = 5
    synapses.I_tau = get_current(taus*1e-3) * amp
    return synapses


def _create_inhibitor_layer_to_output_synapses(inhibitor_layer, output_layer, model_paths):
    equation_builder = SynapseEquationBuilder.import_eq(model_paths.synapse)
    synapses = Connections(
        inhibitor_layer, output_layer, equation_builder=equation_builder, name='inhibitor_layer_to_output_synapses', verbose=False, dt=100*us)
    synapses.connect()
    synapses.weight = -400
    taus = 4
    synapses.I_tau = get_current(taus*1e-3) * amp
    return synapses


class Cache(NamedTuple):
    model_paths: ModelPaths
    neuron_counts: NeuronCount
    hidden_layer: Neurons
    spike_monitors: SpikeMonitors
    interneuron: Optional[Neurons]
    network: Network


def _create_cache(configuration):
    model_paths = load_model_paths()
    neuron_counts = _read_neuron_counts(configuration)
    hidden_layer = _create_non_input_layer(
        model_paths, neuron_counts.hidden, 'hidden')
    number_of_output_neurons = 1
    output_layer = _create_non_input_layer(
        model_paths, number_of_output_neurons, 'output', num_inputs=2)
    hidden_to_output_synapses = _create_hidden_to_output_synapses(
        hidden_layer, output_layer, model_paths, neuron_counts)

    spike_monitors = SpikeMonitors(
        hidden=SpikeMonitor(hidden_layer),
        output=SpikeMonitor(output_layer))
    network = Network(
        hidden_layer,
        spike_monitors.hidden,
        spike_monitors.output,
        output_layer,
        hidden_to_output_synapses)

    interneuron = None
    if configuration.measurement_mode is MeasurementMode.ECOG:
        interneuron = _create_non_input_layer(model_paths, 1, 'interneuron')
        inhibitor_generator = _create_inhibitor_generator()
        inhibitor_layer = _create_non_input_layer(
            model_paths, 1, 'inhibitor', num_inputs=2)
        interneuron_to_inhibitor_synapses = _create_interneuron_to_inhibitor_synapses(
            interneuron, inhibitor_layer, model_paths)
        inhibitor_generator_to_inhibitor_synapses = _create_inhibitor_generator_to_inhibitor_synapses(
            inhibitor_generator, inhibitor_layer, model_paths)
        inhibitor_layer_to_output_synapses = _create_inhibitor_layer_to_output_synapses(
            inhibitor_layer, output_layer, model_paths)
        network.add(
            interneuron,
            inhibitor_generator,
            inhibitor_layer,
            interneuron_to_inhibitor_synapses,
            inhibitor_layer_to_output_synapses,
            inhibitor_generator_to_inhibitor_synapses)

    network.store()

    return Cache(
        model_paths=model_paths,
        neuron_counts=neuron_counts,
        hidden_layer=hidden_layer,
        spike_monitors=spike_monitors,
        interneuron=interneuron,
        network=network
    )


def snn_stage(filtered_spikes, duration, configuration, cache: Cache) -> SpikeMonitors:
    warnings.simplefilter("ignore", DeprecationWarning)
    if cache is None:
        cache = _create_cache(configuration)

    cache.network.restore()

    input_layer = _create_input_layer(
        filtered_spikes, cache.neuron_counts.input)
    input_to_hidden_synapses = _create_input_to_hidden_synapses(
        input_layer,
        cache.hidden_layer,
        cache.model_paths,
        cache.neuron_counts)

    cache.network.add(input_layer)
    cache.network.add(input_to_hidden_synapses)

    if cache.interneuron is not None:
        input_to_interneuron_synapses = _create_input_to_interneuron_synapses(
            input_layer,
            cache.interneuron,
            cache.model_paths)
        cache.network.add(input_to_interneuron_synapses)

    cache.network.run(duration * second)

    cache.network.remove(input_layer)
    cache.network.remove(input_to_hidden_synapses)

    if cache.interneuron is not None:
        cache.network.remove(input_to_interneuron_synapses)

    return cache.spike_monitors
