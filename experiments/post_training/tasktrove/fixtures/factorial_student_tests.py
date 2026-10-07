# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from factorial import calculate_factorial


@pytest.mark.parametrize("number,expected", [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120)])
def test_values(number, expected):
    assert calculate_factorial(number) == expected


def test_negative():
    with pytest.raises(ValueError):
        calculate_factorial(-1)
