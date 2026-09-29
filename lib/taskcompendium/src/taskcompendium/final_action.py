# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode a final assistant message without invoking its advertised functions."""

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

from taskcompendium.models import FunctionCall


@dataclass(frozen=True)
class SubmittedCalls:
    calls: tuple[FunctionCall, ...]


@dataclass(frozen=True)
class SubmittedMessage:
    content: str


SubmittedAction = SubmittedCalls | SubmittedMessage


def unique_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate fields."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


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
        arguments, object_pairs_hook=unique_json_fields, parse_constant=reject_constant, parse_float=parse_float
    )
    if not isinstance(value, dict):
        raise ValueError("Function-call arguments must be a JSON object")
    return value
