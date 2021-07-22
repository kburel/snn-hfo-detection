import pytest
import numpy as np


def are_lists_approximately_equal(first_list, second_list):
    return np.all(first_list == [pytest.approx(_) for _ in second_list])

def are_hfo_detections_equal(first_hfo, second_hfo):
    def are_values_same(key): return are_lists_approximately_equal(
        first_hfo[key], second_hfo[key])
    return first_hfo['total_HFO'] == pytest.approx(
        second_hfo['total_HFO']) and are_values_same('time')and are_values_same('signal') and are_values_same('periods_HFO')