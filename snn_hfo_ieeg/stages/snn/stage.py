import warnings
from brian2.units import second
from snn_hfo_ieeg.stages.snn.cache import Cache, SpikeMonitors, create_cache
from snn_hfo_ieeg.stages.snn.set_input import set_input_spikes, set_advanced_artifact_filter_input_spikes


def snn_stage(filtered_spikes, duration, configuration, cache: Cache) -> SpikeMonitors:
    warnings.simplefilter("ignore", DeprecationWarning)
    if cache is None:
        cache = create_cache(configuration)

    cache.network.restore()

    set_input_spikes(filtered_spikes, cache.input_layer,
                     configuration.measurement_mode)
    if cache.advanced_artifact_filter_input_layer is not None:
        set_advanced_artifact_filter_input_spikes(
            filtered_spikes, cache.advanced_artifact_filter_input_layer)

    cache.network.run(duration * second)

    return cache.spike_monitors
