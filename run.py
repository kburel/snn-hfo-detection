#!/usr/bin/env python3

import numpy as np
from snn_hfo_detection.run import run_hfo_detection
from snn_hfo_detection.user_facing_data import HfoDetectionRun


def _print_hfo(hfo_detection_run: HfoDetectionRun):
    metadata = hfo_detection_run.metadata
    print(
        f'Interval {metadata.interval}, channel # {metadata.channel}: {metadata.channel_label}')
    print(f'SNN simulation will run for {metadata.duration} seconds')
    hfo_detection = hfo_detection_run.detector.run_with_analytics()
    print('Number of HFO events: ', hfo_detection.result.total_amount)
    print('Rate of HFO (event/min)',
          np.around(hfo_detection.result.frequency * 60, decimals=2))
    print('Perios:')
    periods = zip(hfo_detection.analytics.periods.start,
                  hfo_detection.analytics.periods.stop)
    for start, stop in periods:
        print(f'- {start} : {stop}')

    print('------')


if __name__ == '__main__':
    print('Starting HFO detection')
    run_hfo_detection(_print_hfo)
    print('Finished HFO detection')
