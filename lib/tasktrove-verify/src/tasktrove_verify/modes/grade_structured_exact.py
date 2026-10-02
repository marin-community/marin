# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a submitted JSON value with a private reference without scalar coercion."""

import json
from pathlib import Path

from tasktrove_verify.grade import InvalidTask, Reward, read_output, scored
from tasktrove_verify.json_comparison import JsonValue, json_values_equal
from tasktrove_verify.spec import StructuredExactSpec


def validate_structured_exact(spec: StructuredExactSpec) -> None:
    try:
        json.dumps(spec.expected, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise InvalidTask("Structured reference must contain finite JSON values") from error


def grade_structured_exact_candidate(spec: StructuredExactSpec, candidate: JsonValue) -> Reward:
    validate_structured_exact(spec)
    return scored(float(json_values_equal(spec.expected, candidate)))


def grade(spec: StructuredExactSpec, _tests_dir: Path, workspace: Path) -> Reward:
    validate_structured_exact(spec)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    try:
        candidate = json.loads(text)
    except ValueError:
        return scored(0.0, reason="invalid_json")
    return grade_structured_exact_candidate(spec, candidate)
