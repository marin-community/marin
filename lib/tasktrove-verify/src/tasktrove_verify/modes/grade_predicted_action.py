# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare final function calls independently of harness protocol and call IDs."""

import json
import math
from pathlib import Path

from tasktrove_verify.grade import InvalidTask, Reward, read_output, scored
from tasktrove_verify.spec import FunctionCall, PredictedActionSpec, spec_from_table


def _validate_json(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise InvalidTask("Function-call arguments must contain finite JSON values")


def validate_predicted_action(spec: PredictedActionSpec) -> None:
    if not spec.expected_calls:
        raise InvalidTask("Expected function calls are required")
    if spec.numeric_tolerance is not None and (
        type(spec.numeric_tolerance) not in (int, float)
        or not math.isfinite(spec.numeric_tolerance)
        or spec.numeric_tolerance < 0
    ):
        raise InvalidTask("Numeric tolerance must be finite and nonnegative")
    for call in spec.expected_calls:
        if not isinstance(call.name, str) or not call.name or not isinstance(call.arguments, dict):
            raise InvalidTask("Function calls require a nonempty name and argument object")
        _validate_json(call.arguments)


def _arguments_match(expected: object, actual: object, tolerance: float | None) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _arguments_match(value, actual[key], tolerance) for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _arguments_match(left, right, tolerance) for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, float) and isinstance(actual, float) and tolerance is not None:
        return math.isfinite(actual) and abs(expected - actual) <= tolerance
    return expected == actual


def grade_predicted_action_candidate(spec: PredictedActionSpec, actual: tuple[FunctionCall, ...]) -> Reward:
    """Match all calls one to one, preserving JSON types and duplicate multiplicity."""
    validate_predicted_action(spec)
    if len(spec.expected_calls) != len(actual):
        return scored(0.0)
    candidates = [
        [
            index
            for index, right in enumerate(actual)
            if left.name == right.name and _arguments_match(left.arguments, right.arguments, spec.numeric_tolerance)
        ]
        for left in spec.expected_calls
    ]
    matching: dict[int, int] = {}

    def augment(expected_index: int, visited: set[int]) -> bool:
        for actual_index in candidates[expected_index]:
            if actual_index in visited:
                continue
            visited.add(actual_index)
            if actual_index not in matching or augment(matching[actual_index], visited):
                matching[actual_index] = expected_index
                return True
        return False

    for index in sorted(range(len(spec.expected_calls)), key=lambda candidate: len(candidates[candidate])):
        augment(index, set())
    return scored(float(len(matching) == len(spec.expected_calls)))


def grade(spec: PredictedActionSpec, tests_dir: Path, workspace: Path) -> Reward:
    validate_predicted_action(spec)
    candidate = read_output(spec, workspace)
    if candidate is None:
        return scored(0.0, reason="no_output")
    try:
        calls = spec_from_table({"mode": "predicted_action", "expected_calls": json.loads(candidate)})
        assert isinstance(calls, PredictedActionSpec)
        validate_predicted_action(calls)
    except (ValueError, InvalidTask):
        return scored(0.0, reason="invalid_function_calls")
    return grade_predicted_action_candidate(spec, calls.expected_calls)
