# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.json_comparison import NumericTypePolicy, json_values_equal


@pytest.mark.parametrize(
    "expected,actual,equal",
    [
        ({"values": [None, True, 1, 1.0, "1"]}, {"values": [None, True, 1, 1.0, "1"]}, True),
        ({"value": True}, {"value": 1}, False),
        ({"value": 1}, {"value": 1.0}, True),
        ({"value": 1}, {"value": "1"}, False),
        ({"values": [1, 2]}, {"values": [2, 1]}, False),
        ({"values": [1, 2]}, {"values": [1]}, False),
        ({"a": None, "b": [1]}, {"b": [1], "a": None}, True),
        ({"a": None}, {"b": None}, False),
        ({"value": "Answer"}, {"value": "answer"}, False),
    ],
)
def test_json_comparison_preserves_nested_types_keys_and_array_order(expected, actual, equal):
    assert json_values_equal(expected, actual) is equal


@pytest.mark.parametrize(
    "expected,actual,tolerance,equal",
    [
        ({"values": [2.0]}, {"values": [2.125]}, None, False),
        ({"values": [2.0]}, {"values": [2.125]}, 0.125, True),
        ({"values": [2.0]}, {"values": [2.25]}, 0.125, False),
        ({"values": [2]}, {"values": [2.0]}, 0.125, False),
        ({"values": [True]}, {"values": [1]}, 0.125, False),
    ],
)
def test_json_float_tolerance_preserves_scalar_types(expected, actual, tolerance, equal):
    assert json_values_equal(expected, actual, tolerance, numeric_types=NumericTypePolicy.STRICT) is equal


@pytest.mark.parametrize(
    "expected,actual,value_equal,strict_equal",
    [
        ({"values": [16]}, {"values": [16.0]}, True, False),
        ({"values": [16.0]}, {"values": [16]}, True, False),
        ({"value": True}, {"value": 1.0}, False, False),
        ({"value": 2**53 + 1}, {"value": float(2**53)}, False, False),
        ({"value": float(2**53)}, {"value": 2**53 + 1}, False, False),
        ({"value": 10**400}, {"value": 10**400}, True, True),
        ({"value": 10**400}, {"value": 10**400 + 1}, False, False),
    ],
)
def test_json_numeric_policy_preserves_nested_values_and_integer_precision(expected, actual, value_equal, strict_equal):
    assert json_values_equal(expected, actual) is value_equal
    assert json_values_equal(expected, actual, numeric_types=NumericTypePolicy.STRICT) is strict_equal
