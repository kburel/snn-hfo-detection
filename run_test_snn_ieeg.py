import argparse
from typing import List, NamedTuple
from snn_hfo_ieeg.stages.shared_config import Configuration, MeasurementMode
from snn_hfo_ieeg.functions.filter import *
from snn_hfo_ieeg.functions.dynapse_biases import *
from snn_hfo_ieeg.functions.signal_to_spike import *
from snn_hfo_ieeg.functions.hfo_detection import *
from snn_hfo_ieeg.stages.all import run_hfo_detection
from snn_hfo_ieeg.stages.loading.patient_data import load_patient_data, extract_channel_data


class CustomOverrides(NamedTuple):
    duration: float
    channels: List[int]


def _calculate_duration(signal_time):
    extra_simulation_time = 0.050
    return np.max(signal_time) + extra_simulation_time


def run_hfo_detection_for_all_channels(configuration, custom_overrides, hfo_cb):
    # Select Data from a single patient
    patient = 1
    # interval
    current_interval = 1

    patient_data = load_patient_data(
        patient=patient,
        interval=current_interval,
        data_path=configuration.data_path)
    duration = custom_overrides.duration if custom_overrides.duration is not None else _calculate_duration(
        patient_data.signal_time)

    for channel in range(len(patient_data.wideband_signals)):
        if custom_overrides.channels is not None and channel not in custom_overrides.channels:
            continue

        channel_data = extract_channel_data(patient_data, channel)

        print(
            f'Running test for Patient {patient}, interval {current_interval} and channel {channel}')

        print(f'SNN simulation will run for {duration} seconds')
        hfo_detection = run_hfo_detection(
            channel_data=channel_data,
            duration=duration,
            configuration=configuration)

        hfo_count = hfo_detection['total_hfo']

        print('Number of HFO events: ', hfo_count)
        print('Rate of HFO (event/min)',
              np.around((hfo_count/duration)*60, decimals=2))
        print('')

        hfo_cb(hfo_detection)


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Perform an hfo test run')
    default_data_path = 'data/'
    default_hidden_neurons = 86
    parser.add_argument('--data-path', type=str, default=default_data_path,
                        help=f'Specifies the path to the directory containing the test data. Default is {default_data_path}')
    parser.add_argument('--hidden-neurons', type=int, default=default_hidden_neurons,
                        help=f'How many neurons should be in the hidden layer. Default is {default_hidden_neurons}')
    parser.add_argument('--duration', type=float, default=None,
                        help='How many seconds of the dataset should be processed. By default, the entire dataset will be processed')
    parser.add_argument('--channels', type=float, default=None, nargs='*',
                        help='Which channels of the dataset should be processed. By default, all channels will be processed')
    parser.add_argument('mode', type=str,
                        help='Which measurement mode was used to capture the data. Possible values: iEEG, eCoG or scalp.\
                        Note that eCoG will use signals in the fast ripple channel (250-500 Hz), scalp will use the ripple channel (80-250 Hz) and iEEG will use both')
    return parser.parse_args()


def _convert_arguments_to_config(arguments):
    return Configuration(
        data_path=arguments.data_path,
        measurement_mode=MeasurementMode[arguments.mode.upper()],
        hidden_neuron_count=arguments.hidden_neurons
    )


def _convert_arguments_to_custom_overrides(arguments):
    return CustomOverrides(
        duration=arguments.duration,
        channels=arguments.channels)


if __name__ == '__main__':
    arguments = _parse_arguments()
    configuration = _convert_arguments_to_config(arguments)
    custom_overrides = _convert_arguments_to_custom_overrides(arguments)
    run_hfo_detection_for_all_channels(
        configuration=configuration,
        custom_overrides=custom_overrides,
        hfo_cb=lambda _: {})
