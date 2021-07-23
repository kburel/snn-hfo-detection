import argparse
from snn_hfo_ieeg.functions.filter import *
from snn_hfo_ieeg.functions.dynapse_biases import *
from snn_hfo_ieeg.functions.signal_to_spike import *
from snn_hfo_ieeg.functions.hfo_detection import *
from snn_hfo_ieeg.stages.all import run_hfo_detection
from snn_hfo_ieeg.stages.loading.network_parameters import load_network_parameters
from snn_hfo_ieeg.stages.loading.patient_data import load_patient_data, extract_channel_data


def _calculate_duration(signal_time):
    extra_simulation_time = 0.050
    return np.max(signal_time) + extra_simulation_time


def run_hfo_detection_for_all_channels(data_path, hfo_cb):
    # Select Data from a single patient
    patient = 1
    # interval
    current_interval = 1

    patient_data = load_patient_data(
        patient=patient,
        interval=current_interval,
        data_path=data_path)
    duration = _calculate_duration(patient_data.signal_time)

    network_parameters = load_network_parameters()

    for channel in range(len(patient_data.wideband_signals)):
        channel_data = extract_channel_data(patient_data, channel)

        print(
            f'Running test for Patient {patient}, interval {current_interval} and channel {channel}')

        print('SNN simulation will run for ', duration, ' seconds')
        hfo_detection = run_hfo_detection(
            channel_data,
            duration=duration,
            network_parameters=network_parameters)

        hfo_count = hfo_detection['total_hfo']

        print('Number of HFO events: ', hfo_count)
        print('Rate of HFO (event/min)',
              np.around((hfo_count/duration)*60, decimals=2))
        print('')

        hfo_cb(hfo_detection)


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Perform an hfo test run')
    parser.add_argument('--data-path', type=str, nargs='?', default='data/',
                        help='Specifies the path to the directory containing the test data. Default is ./data/')
    return parser.parse_args()


if __name__ == '__main__':
    data_path = _parse_arguments().data_path
    run_hfo_detection_for_all_channels(data_path, lambda _: {})
