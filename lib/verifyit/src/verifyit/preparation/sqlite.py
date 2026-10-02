# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared finite SQLite row-set policy for literal Exact inputs."""

import json
import math
from fractions import Fraction

SQLiteValue = int | float | str | bytes | None


def prepare_sqlite_row_set(rows: tuple[tuple[SQLiteValue, ...], ...]) -> str:
    """Encode finite SQLite rows with Python numeric equality and set semantics.

    This unsafe preparation discards row order, multiplicity and the distinction
    between equal integer/float values (including signed zero). Column positions
    within each row remain significant; column names and empty-result column
    counts are outside this contract. Text, bytes and null remain distinct.

    Callers retain raw observations and select a named source policy before
    invoking this function. It does not compare candidate and reference values
    or grade. Nonfinite numbers and values outside SQLite's default scalar types
    raise ValueError; callers retain task/candidate/infrastructure identity.
    """
    encoded = []
    for row in rows:
        values: list[list[str | None]] = []
        for value in row:
            if type(value) not in (int, float, str, bytes, type(None)):
                raise ValueError("SQLite row-set policy requires default SQLite scalar types")
            if isinstance(value, (int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("SQLite row-set policy requires finite numbers")
                number = Fraction(value)
                values.append(["number", str(number.numerator), str(number.denominator)])
            elif type(value) is bytes:
                values.append(["bytes", value.hex()])
            elif value is None or type(value) is str:
                values.append([type(value).__name__, value])
            else:
                raise ValueError("SQLite row-set policy requires default SQLite scalar types")
        encoded.append(json.dumps(values, ensure_ascii=False, separators=(",", ":")))
    return json.dumps(sorted(set(encoded)), ensure_ascii=False, separators=(",", ":"))
