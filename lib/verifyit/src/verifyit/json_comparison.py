# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare JSON values with an explicit numeric type policy."""

from enum import StrEnum
from typing import TypeAlias

# The verifier package supports Python 3.11, before the type statement was introduced.
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]  # noqa: UP040


class NumericTypePolicy(StrEnum):
    VALUE = "value"
    STRICT = "strict"


def json_values_equal(
    expected: JsonValue,
    actual: JsonValue,
    numeric_tolerance: float | None = None,
    *,
    numeric_types: NumericTypePolicy = NumericTypePolicy.VALUE,
) -> bool:
    """Compare ordered JSON values; strict policy also distinguishes integers from floats."""
    if type(expected) is not type(actual):
        if numeric_types is NumericTypePolicy.VALUE and type(expected) in (int, float) and type(actual) in (int, float):
            return expected == actual
        return False
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            json_values_equal(value, actual[key], numeric_tolerance, numeric_types=numeric_types)
            for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            json_values_equal(left, right, numeric_tolerance, numeric_types=numeric_types)
            for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, float) and isinstance(actual, float) and numeric_tolerance is not None:
        return abs(expected - actual) <= numeric_tolerance
    return expected == actual
