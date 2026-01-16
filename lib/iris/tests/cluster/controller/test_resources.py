# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for resource parsing and comparison utilities."""

import pytest

from iris.cluster.controller.resources import (
    parse_memory_string,
)


@pytest.mark.parametrize(
    "memory_str,expected_bytes",
    [
        ("1g", 1024**3),
        ("8g", 8 * 1024**3),
        ("16gb", 16 * 1024**3),
        ("512m", 512 * 1024**2),
        ("1024mb", 1024 * 1024**2),
        ("1024k", 1024 * 1024),
        ("1024kb", 1024 * 1024),
        ("1024b", 1024),
        ("1024", 1024),  # No unit defaults to bytes
        ("", 0),
        ("0g", 0),
    ],
)
def test_parse_memory_string(memory_str, expected_bytes):
    assert parse_memory_string(memory_str) == expected_bytes


def test_parse_memory_string_case_insensitive():
    assert parse_memory_string("8G") == parse_memory_string("8g")
    assert parse_memory_string("16GB") == parse_memory_string("16gb")
    assert parse_memory_string("512M") == parse_memory_string("512m")


def test_parse_memory_string_with_whitespace():
    assert parse_memory_string("  8g  ") == 8 * 1024**3


def test_parse_memory_string_invalid():
    with pytest.raises(ValueError):
        parse_memory_string("invalid")
    with pytest.raises(ValueError):
        parse_memory_string("8x")
