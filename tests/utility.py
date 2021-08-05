import pathlib
import pytest
import numpy as np


def assert_are_lists_approximately_equal(first_list, second_list, accuracy=None):
    assert np.all(np.array(first_list, dtype=np.array(second_list).dtype) == [
                  pytest.approx(_, abs=accuracy) for _ in second_list])


def assert_are_hfo_detections_equal(first_hfo, second_hfo):
    def assert_are_values_same(property_cb):
        assert_are_lists_approximately_equal(
            property_cb(first_hfo), property_cb(second_hfo))
    assert first_hfo.result.total_amount == second_hfo.result.total_amount
    assert first_hfo.result.frequency == pytest.approx(
        second_hfo.result.frequency)
    assert_are_values_same(lambda hfo: hfo.analytics.detections)
    assert_are_values_same(lambda hfo: hfo.analytics.periods.start)
    assert_are_values_same(lambda hfo: hfo.analytics.periods.stop)


def get_tests_path():
    return pathlib.Path(__file__).parent.resolve()
