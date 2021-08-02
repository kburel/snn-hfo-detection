import argparse
import sys
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode
from snn_hfo_ieeg.entrypoint.hfo_detection import CustomOverrides
from snn_hfo_ieeg.stages.plotting.plot_loader import find_plotting_functions


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform an hfo test run')
    default_data_path = 'data/'
    default_hidden_neurons = 86
    parser.add_argument('mode', type=str,
                        help='Which measurement mode was used to capture the data. Possible values: iEEG, eCoG or scalp.\
                        Note that eCoG will use signals in the fast ripple channel (250-500 Hz), scalp will use the ripple channel (80-250 Hz) and iEEG will use both')
    parser.add_argument('--data-path', type=str, default=default_data_path,
                        help=f'Specifies the path to the directory containing the test data. Default is {default_data_path}')
    parser.add_argument('--hidden-neurons', type=int, default=default_hidden_neurons,
                        help=f'How many neurons should be in the hidden layer. Default is {default_hidden_neurons}')
    parser.add_argument('--duration', type=float, default=None,
                        help='How many seconds of the dataset should be processed. By default, the entire dataset will be processed')
    parser.add_argument('--channels', type=int, default=None, nargs='+',
                        help='Which channels of the dataset should be processed, using 1 based indexing. By default, all channels will be processed')
    parser.add_argument('--patients', type=int, default=None, nargs='+',
                        help='Which patients should be processed. By default, all patients will be processed')
    parser.add_argument('--intervals', type=int, default=None, nargs='+',
                        help='Which intervals should be processed. By default, all intervals will be processed. Only works when --patients was called beforehand with exactly one patient number.')
    parser.add_argument('--plot', type=str, default=None, nargs='+',
                        help='Which plots should be generated during the HFO detection. Possible values: raster')
    return parser.parse_args()


def _get_selected_plots(plot_names):
    try:
        return find_plotting_functions(plot_names)
    except ValueError as error:
        sys.exit(f'run.py: error: {error}')


def convert_arguments_to_config(arguments):
    return Configuration(
        data_path=arguments.data_path,
        measurement_mode=MeasurementMode[arguments.mode.upper()],
        hidden_neuron_count=arguments.hidden_neurons,
        plots=_get_selected_plots(arguments.plot)
    )


def _validate_custom_overrides(custom_overrides):
    if custom_overrides.patients is None and custom_overrides.intervals is not None:
        sys.exit(
            'run.py: error: --intervals requires --patients with exactly one patient, but you did not specify any')
    if custom_overrides.patients is not None and len(custom_overrides.patients) != 1 and custom_overrides.intervals is not None:
        sys.exit(
            'run.py: error: --intervals requires --patients with exactly one patient, but you did specified more')


def convert_arguments_to_custom_overrides(arguments):
    custom_overrides = CustomOverrides(
        duration=arguments.duration,
        channels=arguments.channels,
        patients=arguments.patients,
        intervals=arguments.intervals)
    _validate_custom_overrides(custom_overrides)
    return custom_overrides
