# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare JSON values without coercing scalar types."""

from typing import TypeAlias

# The verifier package supports Python 3.11, before the type statement was introduced.
JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None  # noqa: UP040


def json_values_equal(expected: JsonValue, actual: JsonValue, numeric_tolerance: float | None = None) -> bool:
    """Compare JSON types and ordered arrays, optionally tolerating float differences."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            json_values_equal(value, actual[key], numeric_tolerance) for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            json_values_equal(left, right, numeric_tolerance) for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, float) and isinstance(actual, float) and numeric_tolerance is not None:
        return abs(expected - actual) <= numeric_tolerance
    return expected == actual
