# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Mode numeric: the last number the candidate wrote, within absolute or relative tolerance.

The number comes from the last ``\boxed{}`` expression when the output has one, otherwise from the
whole output; in either case the last number wins, so a candidate that shows its work and states a
result is scored on the result. Thousands separators are dropped and exponent notation is read.
A candidate matches when it is within ``max(tolerance_abs, tolerance_rel * |expected|)``.
"""

import math
import re
from pathlib import Path

from tasktrove_verify.modes.extract import extract_boxed
from tasktrove_verify.output import read_output
from tasktrove_verify.reward import InvalidTask, Reward, scored
from tasktrove_verify.spec import NumericSpec

NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?")


def _last_number(text: str) -> float | None:
    boxed = extract_boxed(text)
    sources = [boxed, text] if boxed else [text]
    for source in sources:
        matches = NUMBER.findall(source)
        if matches:
            return float(matches[-1].replace(",", ""))
    return None


def grade(spec: NumericSpec, tests_dir: Path, workspace: Path) -> Reward:
    if not math.isfinite(spec.expected):
        raise InvalidTask(f"numeric expected must be a finite number, got {spec.expected}")

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    value = _last_number(text)
    if value is None:
        return scored(0.0, reason="no_number", expected=spec.expected)
    tolerance = max(spec.tolerance_abs, spec.tolerance_rel * abs(spec.expected))
    match = abs(value - spec.expected) <= tolerance
    return scored(float(match), extracted=value, expected=spec.expected, tolerance=tolerance)
