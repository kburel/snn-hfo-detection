from functools import reduce
from brian2 import SpikeGeneratorGroup
from brian2.units import us, second
import numpy as np
from snn_hfo_ieeg.functions.signal_to_spike import concatenate_spikes
from snn_hfo_ieeg.stages.snn.tau_generation import generate_concatenated_taus
from snn_hfo_ieeg.stages.snn.weight_generation import generate_weights
from snn_hfo_ieeg.stages.snn.creation import create_synapses


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


def create_input_to_hidden_synapses(input_layer, hidden_layer, model_paths, neuron_counts):
    weights = generate_weights(neuron_counts)
    taus = generate_concatenated_taus(neuron_counts)
    return create_synapses(
        'input_to_hidden', model_paths, input_layer, hidden_layer, weights, taus)


def create_hidden_to_output_synapses(hidden_layer, output_layer, model_paths, hidden_neuron_count):
    weights = np.repeat(3_000.0, hidden_neuron_count.hidden)
    taus = np.repeat(10, hidden_neuron_count.hidden)
    return create_synapses(
        'hidden_to_output', model_paths, hidden_layer, output_layer, weights, taus)
