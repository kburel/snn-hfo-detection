import pytest
import numpy as np


def are_lists_approximately_equal(first_list, second_list):
    return np.all(first_list == [pytest.approx(_) for _ in second_list])
