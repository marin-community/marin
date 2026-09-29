# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare predicted function calls without dispatching them."""

from typing import Any

from taskcompendium.final_action import SubmittedAction, SubmittedMessage, parse_arguments
from taskcompendium.models import FunctionCall, ToolCallComparatorConfig


def _arguments_match(expected: Any, actual: Any, config: ToolCallComparatorConfig) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _arguments_match(value, actual[key], config) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            _arguments_match(left, right, config) for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, float) and config.numeric_tolerance is not None:
        return abs(expected - actual) <= config.numeric_tolerance
    return expected == actual


def _call_matches(expected: FunctionCall, actual: FunctionCall, config: ToolCallComparatorConfig) -> bool:
    if expected.name != actual.name:
        return False
    try:
        expected_arguments = parse_arguments(expected.arguments)
        actual_arguments = parse_arguments(actual.arguments)
        return _arguments_match(expected_arguments, actual_arguments, config)
    except (ValueError, TypeError):
        return False


def _matching_count(
    expected: tuple[FunctionCall, ...], actual: tuple[FunctionCall, ...], config: ToolCallComparatorConfig
) -> int:
    candidates = [
        [index for index, right in enumerate(actual) if _call_matches(left, right, config)] for left in expected
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

    for index in sorted(range(len(expected)), key=lambda candidate: len(candidates[candidate])):
        augment(index, set())
    return len(matching)


def compare(expected: tuple[FunctionCall, ...], actual: SubmittedAction, config: ToolCallComparatorConfig) -> float:
    """Require the same number of calls with exact JSON arguments by default."""
    if not expected:
        raise ValueError("Expected function calls are required")
    if isinstance(actual, SubmittedMessage):
        return 0.0
    if len(expected) != len(actual.calls):
        return 0.0
    matched = _matching_count(expected, actual.calls, config)
    return float(matched == len(expected))
