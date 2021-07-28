#!/usr/bin/env python3

from snn_hfo_ieeg.run import run_hfo_detection


def _print_hfo(hfo_detection_run):
    print(
        f'Results for patient {hfo_detection_run.patient}, interval {hfo_detection_run.interval} and channel {hfo_detection_run.channel}')
    print(f'SNN simulation will run for {hfo_detection_run.duration} seconds')
    print('Number of HFO events: ', hfo_detection_run.hfo_detection.total_amount)
    print('Rate of HFO (event/min)',
          hfo_detection_run.hfo_detection.frequency * 60, decimals=2)


if __name__ == '__main__':
    run_hfo_detection(_print_hfo)
