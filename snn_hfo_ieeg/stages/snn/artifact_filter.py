import numpy as np
from brian2.input.poissongroup import PoissonGroup
from brian2.units import us, Hz
from snn_hfo_ieeg.stages.snn.creation import create_synapses, create_non_input_layer
from snn_hfo_ieeg.user_facing_data import MeasurementMode


def _create_interneuron_to_inhibitor_synapses(interneuron_layer, inhibitor_layer, model_paths):
    weights = np.array([-10_000])
    taus = np.array([20])
    return create_synapses(
        'interneuron_to_inhibitor', model_paths, interneuron_layer, inhibitor_layer, weights, taus)


def _create_inhibitor_generator_to_inhibitor_synapses(inhibitor_generator, inhibitor_layer, model_paths):
    weights = np.array([50_000])
    taus = np.array([5])
    return create_synapses(
        'inhibitor_generator_to_inhibitor_layer', model_paths, inhibitor_generator, inhibitor_layer, weights, taus)


def _create_inhibitor_layer_to_output_synapses(inhibitor_layer, output_layer, model_paths):
    weights = np.array([-10_000])
    taus = np.array([10])
    return create_synapses(
        'inhibitor_layer_to_output', model_paths, inhibitor_layer, output_layer, weights, taus)


def _create_inhibitor_generator():
    return PoissonGroup(1, 135*Hz, name='inhibitor_generator', dt=100*us)


def add_artifact_filter_to_network_and_get_interneuron(model_paths, output_layer, network):
    interneuron = create_non_input_layer(model_paths, 1, 'interneuron')
    inhibitor_generator = _create_inhibitor_generator()
    inhibitor_layer = create_non_input_layer(
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


def _create_input_to_interneuron_synapses(input_layer, interneuron_layer, model_paths):
    weights = np.array([2_000])
    taus = np.array([5])
    return create_synapses(
        'input_to_interneuron', model_paths, input_layer, interneuron_layer, weights, taus)


def add_input_to_artifact_filter_to_network(input_layer, cache):
    input_to_interneuron_synapses = _create_input_to_interneuron_synapses(
        input_layer,
        cache.interneuron,
        cache.model_paths)
    cache.network.add(input_to_interneuron_synapses)
    return input_to_interneuron_synapses


def should_add_artifact_filter(configuration):
    return configuration.measurement_mode is MeasurementMode.ECOG or configuration.measurement_mode is MeasurementMode.SCALP
