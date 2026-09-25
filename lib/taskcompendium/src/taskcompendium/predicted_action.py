# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare final NeMo actions without dispatching the advertised functions."""

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

from taskcompendium.models import FunctionCall, ToolCallComparatorConfig
from taskcompendium.submission import _object_with_unique_fields


@dataclass(frozen=True)
class SubmittedCalls:
    calls: tuple[FunctionCall, ...]


@dataclass(frozen=True)
class SubmittedMessage:
    content: str


SubmittedAction = SubmittedCalls | SubmittedMessage


def decode_action(message: dict[str, Any]) -> SubmittedAction:
    """Decode one native assistant message; malformed envelopes are submission errors."""
    if message.get("role") != "assistant":
        raise ValueError("Final action requires an assistant message")
    raw_calls = message.get("tool_calls")
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise ValueError("Final tool_calls must be a list")
        calls = []
        for call in raw_calls:
            function = call.get("function") if isinstance(call, dict) and call.get("type") == "function" else None
            if not isinstance(function, dict):
                raise ValueError("Final action requires native function calls")
            name, arguments = function.get("name"), function.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, str):
                raise ValueError("Final function calls require a name and argument string")
            calls.append(FunctionCall(name, arguments))
        if calls:
            return SubmittedCalls(tuple(calls))
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Final action requires a message or function call")
    return SubmittedMessage(content)


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


def parse_arguments(arguments: str) -> dict[str, Any]:
    """Decode a native function's JSON object without accepting duplicate keys."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON argument: {value}")

    def parse_float(value: str) -> float:
        number = float(value)
        if not isfinite(number):
            raise ValueError("Non-finite JSON argument")
        return number

    value = json.loads(
        arguments, object_pairs_hook=_object_with_unique_fields, parse_constant=reject_constant, parse_float=parse_float
    )
    if not isinstance(value, dict):
        raise ValueError("Function-call arguments must be a JSON object")
    return value


def _call_matches(expected: FunctionCall, actual: FunctionCall, config: ToolCallComparatorConfig) -> bool:
    if expected.name != actual.name:
        return False
    try:
        expected_arguments = parse_arguments(expected.arguments)
        actual_arguments = parse_arguments(actual.arguments)
        return _arguments_match(expected_arguments, actual_arguments, config)
    except (json.JSONDecodeError, ValueError):
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
