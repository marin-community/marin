# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct port of NeMo Gym's single-step tool-call comparison reward."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any


class StepRewardCategory(StrEnum):
    NO_ACTION_FOUND = "No tool call or chat message was found in the response"
    NO_EXPECTED_TOOL_CALL = "No tool call was found when one was expected"
    EXPECTED_CHAT_MESSAGE_FOUND = "A chat message was found as expected"
    NO_EXPECTED_CHAT_MESSAGE = "A tool call was executed when a chat message was expected"
    UNEXPECTED_TOOL = "The tool in a tool call is not the expected tool"
    ARGUMENTS_DECODE_ERROR = "An error occurred when decoding the arguments string in a tool call as a JSON object"
    ARGUMENT_VALUE_TYPE_DIFFERENT = "The type of an argument value in a tool call is different than the expected type"
    ARGUMENT_OBJECT_KEYS_DIFFERENT = (
        "The keys in an object in an argument value in a tool call are different than the keys in the expected object"
    )
    ARGUMENT_LIST_LENGTH_DIFFERENT = (
        "A list in an argument value in a tool call has a different length than the expected list"
    )
    ARGUMENT_VALUE_DIFFERENT = "An argument value in a tool call is different than the expected value"
    EXPECTED_TOOL_CALL = "A tool call that matches the expected tool call was found"


def _compare_arguments(
    expected: Any,
    actual: Any,
    *,
    floating_point_comparison_threshold: float = 1e-6,
) -> tuple[bool, StepRewardCategory | None]:
    if not isinstance(actual, type(expected)):
        return False, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return False, StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT
        for key, value in expected.items():
            matches, category = _compare_arguments(
                value,
                actual[key],
                floating_point_comparison_threshold=floating_point_comparison_threshold,
            )
            if not matches:
                return matches, category
        return True, None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False, StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT
        for expected_item, actual_item in zip(expected, actual):
            matches, category = _compare_arguments(
                expected_item,
                actual_item,
                floating_point_comparison_threshold=floating_point_comparison_threshold,
            )
            if not matches:
                return matches, category
        return True, None
    if isinstance(expected, float):
        if abs(actual - expected) < floating_point_comparison_threshold:
            return True, None
        return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
    if expected == actual:
        return True, None
    return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT


def grade_expected_action(
    expected_action: dict[str, Any],
    assistant_message: dict[str, Any],
) -> tuple[float, StepRewardCategory]:
    """Return the exact binary reward and diagnostic category used by NeMo Gym."""
    tool_calls = assistant_message.get("tool_calls") or []
    content = assistant_message.get("content")
    expected_type = expected_action.get("type")
    if expected_type == "message":
        if isinstance(content, str) and content.strip() and not tool_calls:
            return 1.0, StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND
        return 0.0, StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE
    if expected_type != "function_call":
        raise NotImplementedError(f"Unsupported expected action type {expected_type!r}")
    if not tool_calls:
        return 0.0, (
            StepRewardCategory.NO_EXPECTED_TOOL_CALL if isinstance(content, str) else StepRewardCategory.NO_ACTION_FOUND
        )

    if len(tool_calls) != 1:
        return 0.0, StepRewardCategory.NO_EXPECTED_TOOL_CALL

    actual = tool_calls[0].get("function") or {}
    if expected_action.get("name") != actual.get("name"):
        return 0.0, StepRewardCategory.UNEXPECTED_TOOL
    try:
        expected_arguments = json.loads(expected_action["arguments"])
        actual_arguments = json.loads(actual.get("arguments"))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return 0.0, StepRewardCategory.ARGUMENTS_DECODE_ERROR
    matches, category = _compare_arguments(
        expected_arguments,
        actual_arguments,
    )
    if matches:
        return 1.0, StepRewardCategory.EXPECTED_TOOL_CALL
    assert category is not None
    return 0.0, category
