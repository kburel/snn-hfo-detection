from typing import NamedTuple
import numpy as np


class Quarters(NamedTuple):
    first: np.array
    second: np.array
    third: np.array
    fourth: np.array


def quarter(array):
    if len(array) % 4 != 0 or len(array) < 4:
        raise ValueError(
            f'Length of array must be a positive multiple of four. Actual length: {len(array)}')
    quarter_point = len(array) // 4
    first_quarter = array[:quarter_point]
    second_quarter = array[quarter_point:2*quarter_point]
    third_quarter = array[2*quarter_point:3*quarter_point]
    fourth_quarter = array[3*quarter_point:]
    return Quarters(
        first=first_quarter,
        second=second_quarter,
        third=third_quarter,
        fourth=fourth_quarter)
