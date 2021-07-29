import pathlib
import pytest
import numpy as np


def are_lists_approximately_equal(first_list, second_list):
    return np.all(first_list == [pytest.approx(_) for _ in second_list])


def are_hfo_detections_equal(first_hfo, second_hfo):
    def are_values_same(property_cb):
        return are_lists_approximately_equal(
            property_cb(first_hfo), property_cb(second_hfo))
    return first_hfo.result.total_amount == second_hfo.result.total_amount \
        and first_hfo.result.frequency == pytest.approx(second_hfo.result.frequency) \
        and are_values_same(lambda hfo: hfo.analytics.analyzed_times) \
        and are_values_same(lambda hfo: hfo.analytics.detections) \
        and are_values_same(lambda hfo: hfo.analytics.periods.start) \
        and are_values_same(lambda hfo: hfo.analytics.periods.stop)


def get_tests_path():
    return pathlib.Path(__file__).parent.resolve()
