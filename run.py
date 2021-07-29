#!/usr/bin/env python3

import numpy as np
from snn_hfo_ieeg.run import run_hfo_detection
from snn_hfo_ieeg.entrypoint.hfo_detection import HfoDetector, Metadata


def _print_hfo(metadata: Metadata, hfo_detector: HfoDetector):
    print(
        f'Patient {metadata.patient}, interval {metadata.interval}, channel {metadata.channel}')
    print(f'SNN simulation will run for {metadata.duration} seconds')
    hfo_detection = hfo_detector.run()
    print('Number of HFO events: ', hfo_detection.total_amount)
    print('Rate of HFO (event/min)',
          np.around(hfo_detection.frequency * 60, decimals=2))
    print('------')


if __name__ == '__main__':
    print('Starting HFO detection')
    run_hfo_detection(_print_hfo)
    print('Finished HFO detection')
