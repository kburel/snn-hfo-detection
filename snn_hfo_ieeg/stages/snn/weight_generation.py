import random
import numpy as np
from snn_hfo_ieeg.stages.snn.concatenation import concatenate_excitatory_and_inhibitory_with_generator_function
POSSIBLE_ABSOLUTE_WEIGHTS = [1000, 2000]


def _generate_weights(hidden_neuron_count):
    excitatory_weights = np.array(random.choices(POSSIBLE_ABSOLUTE_WEIGHTS,
                                                 k=int(hidden_neuron_count)))
    inhibitory_weights = excitatory_weights * -1
    return excitatory_weights, inhibitory_weights


def generate_weights(input_neuron_count, hidden_neuron_count):
    return concatenate_excitatory_and_inhibitory_with_generator_function(input_neuron_count, hidden_neuron_count, _generate_weights)
