#!/usr/bin/env python3

import numpy as np
from snn_hfo_ieeg.run import run_hfo_detection
from snn_hfo_ieeg.entrypoint.hfo_detection import HfoDetectionRun


def _print_hfo(hfo_detection_run: HfoDetectionRun):
    print(
        f'Results for patient {hfo_detection_run.patient}, interval {hfo_detection_run.interval} and channel {hfo_detection_run.channel}')
    print(f'SNN simulation ran for {hfo_detection_run.duration} seconds')
    print('Number of HFO events: ', hfo_detection_run.hfo_detection.total_amount)
    print('Rate of HFO (event/min)',
          np.around(hfo_detection_run.hfo_detection.frequency * 60, decimals=2))
    print('------')


if __name__ == '__main__':
    print('Starting HFO detection')
    run_hfo_detection(_print_hfo)
    print('Finished HFO detection')
