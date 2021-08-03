import argparse
import sys
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode, PlotMode
from snn_hfo_ieeg.entrypoint.hfo_detection import CustomOverrides
from snn_hfo_ieeg.plotting.plot_loader import find_plotting_functions


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform an hfo test run')
    default_data_path = 'data/'
    default_saving_path = 'saved_data/'
    default_plot_path = 'plots/'
    default_plot_mode = PlotMode.SAVE.name
    default_hidden_neurons = 86
    default_calibration = 10
    parser.add_argument('mode', type=str,
                        help='Which measurement mode was used to capture the data. Possible values: iEEG, eCoG or scalp.\
                        Note that eCoG will use signals in the fast ripple channel (250-500 Hz), scalp will use the ripple channel (80-250 Hz) and iEEG will use both')
    parser.add_argument('--data-path', type=str, default=default_data_path,
                        help=f'Specifies the path to the directory containing the test data. Default is {default_data_path}')
    parser.add_argument('--hidden-neurons', type=int, default=default_hidden_neurons,
                        help=f'How many neurons should be in the hidden layer. Default is {default_hidden_neurons}. Ignored when loading data with --load')
    parser.add_argument('--duration', type=float, default=None,
                        help='How many seconds of the dataset should be processed. By default, the entire dataset will be processed. Ignored when loading data with --load')

    parser.add_argument('--disable-saving', action='store_true',
                        help='Disables HFO detections saving. By default, all HFO detections are saved to the path specified by --save')
    parser.add_argument('--calibration', type=float, default=default_calibration,
                        help=f'How many seconds of the dataset should be used for calibration of HFO thresholds. Default is {default_calibration} s. If calibration is bigger than duration, the entire duration will be used for calibration. Ignored when loading data with --load')
    parser.add_argument('--channels', type=int, default=None, nargs='+',
                        help='Which channels of the dataset should be processed, using 1 based indexing. By default, all channels will be processed')
    parser.add_argument('--patients', type=int, default=None, nargs='+',
                        help='Which patients should be processed. By default, all patients will be processed')
    parser.add_argument('--intervals', type=int, default=None, nargs='+',
                        help='Which intervals should be processed. By default, all intervals will be processed. Only works when --patients was called beforehand with exactly one patient number.')
    parser.add_argument('--plot', type=str, default=[], nargs='+',
                        help='Which plots should be generated during the HFO detection. Possible values: raster')
    parser.add_argument('--plot-mode', type=str, default=default_plot_mode,
                        help=f'How to handle plots. Possible values: save, show, both. Default is {default_plot_mode}')
    parser.add_argument('--plot-path', type=str, default=default_plot_path,
                        help=f'Location to save plots to when --plot-mode is set to "save". Default is {default_plot_path}')

    persistence_group = parser.add_mutually_exclusive_group()
    persistence_group.add_argument('--save', type=str, default=default_saving_path,
                                   help=f'Path to where the HFO detections should be saved. Default is {default_saving_path} if --load was not specified.')
    persistence_group.add_argument('--load', type=str, default=None, nargs='?', const=default_saving_path,
                                   help=f'Path to where the HFO detections where saved with --save. By default, no previously saved data will be loaded. If --load was specified with no path, {default_saving_path} will by used.')

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
        calibration_time=arguments.calibration,
        plots=_get_selected_plots(arguments.plot),
        saving_path=arguments.save,
        disable_saving=arguments.disable_saving,
        loading_path=arguments.load,
        plot_path=arguments.plot_path,
        plot_mode=PlotMode[arguments.plot_mode.upper()],
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
