import re
import os
from typing import NamedTuple, TypedDict

_INTERVAL_REGEX = re.compile(r'^.*I(\d)\.mat$')


class Intervals(TypedDict):
    index: int
    path: str


class _Match(NamedTuple):
    path: str
    index: int


def _get_file_and_directory_paths(data_path):
    return [os.path.join(data_path, filename)
            for filename in os.listdir(data_path)]


def _get_files(data_path):
    return [file_or_directory for file_or_directory
            in _get_file_and_directory_paths(data_path)
            if not os.path.isdir(file_or_directory)]


def _parse_match(match):
    path = match.string
    index = int(match.groups()[0])
    return _Match(path, index)


def _filter_paths(paths, regex):
    regex_matches = [regex.match(path) for path in paths]
    return [_parse_match(match) for match in regex_matches if match]


def _get_files_of_regex(data_path, regex):
    files = _get_files(data_path)
    return _filter_paths(files, regex)


def _convert_matches_to_intervals(matches):
    return {interval.index: interval.path for interval in matches}


def get_interval_paths(data_path: str) -> Intervals:
    interval_matches = _get_files_of_regex(data_path, _INTERVAL_REGEX)
    return _convert_matches_to_intervals(interval_matches)
