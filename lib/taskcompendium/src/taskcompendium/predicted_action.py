# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned NeMo next-action extraction and comparison without action dispatch."""

import json
from collections import Counter
from typing import Any

from taskcompendium.models import (
    ExpectedAction,
    ExpectedFunctionCall,
    ExpectedFunctionCallBatch,
    ExpectedMessage,
    FunctionCall,
    ParallelToolCallRewardMode,
    ToolCallComparatorConfig,
)


def action_from_transcript(transcript: tuple[dict[str, Any], ...]) -> ExpectedAction | None:
    """Read the final native assistant output, preserving source tool-call precedence."""
    assistants = [item for item in transcript if item.get("role") == "assistant"]
    if not assistants:
        return None
    final = assistants[-1]
    raw_calls = final.get("tool_calls", ())
    if isinstance(raw_calls, list):
        calls: list[FunctionCall] = []
        for call in raw_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                return None
            name, arguments = function.get("name"), function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, str):
                return None
            calls.append(FunctionCall(name, arguments))
        if len(calls) == 1:
            return ExpectedFunctionCall(calls[0].name, calls[0].arguments)
        if calls:
            return ExpectedFunctionCallBatch(tuple(calls))
    content = final.get("content")
    return ExpectedMessage(content) if isinstance(content, str) else None


def _calls(action: ExpectedAction) -> tuple[FunctionCall, ...]:
    if isinstance(action, ExpectedFunctionCall):
        return (FunctionCall(action.name, action.arguments),)
    if isinstance(action, ExpectedFunctionCallBatch):
        return action.calls
    return ()


def _tool_call_matches(
    expected: FunctionCall, actual: FunctionCall, config: ToolCallComparatorConfig
) -> tuple[bool, str]:
    if expected.name != actual.name:
        return False, "unexpected_tool"
    try:
        expected_arguments = json.loads(expected.arguments)
        actual_arguments = json.loads(actual.arguments)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, "arguments_decode_error"
    return _argument_matches(expected_arguments, actual_arguments, config)


def _argument_matches(expected: Any, actual: Any, config: ToolCallComparatorConfig) -> tuple[bool, str]:
    # Preserve the pinned comparator's Python `isinstance` behavior: because
    # bool subclasses int, JSON true is accepted where the expected value is 1.
    if not isinstance(actual, type(expected)):
        return False, "argument_value_type_different"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return False, "argument_object_keys_different"
        for key, value in expected.items():
            matched, category = _argument_matches(value, actual[key], config)
            if not matched:
                return matched, category
        return True, "expected_tool_call"
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False, "argument_list_length_different"
        for expected_value, actual_value in zip(expected, actual, strict=True):
            matched, category = _argument_matches(expected_value, actual_value, config)
            if not matched:
                return matched, category
        return True, "expected_tool_call"
    if isinstance(expected, float):
        return (
            (True, "expected_tool_call")
            if abs(actual - expected) < config.floating_point_comparison_threshold
            else (False, "argument_value_different")
        )
    if isinstance(expected, str):
        expected_counts = Counter(expected.strip().lower().split())
        actual_counts = Counter(actual.strip().lower().split())
        if expected_counts.total() < 2 or actual_counts.total() < 2:
            return (True, "expected_tool_call") if expected == actual else (False, "argument_value_different")
        similarity = (expected_counts & actual_counts).total() / (expected_counts.total() + actual_counts.total())
        return (
            (True, "expected_tool_call")
            if similarity >= config.word_count_similarity_threshold
            else (False, "argument_value_different")
        )
    return (True, "expected_tool_call") if expected == actual else (False, "argument_value_different")


def _maximum_matching(candidates: list[list[int]]) -> dict[int, int]:
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

    for expected_index in sorted(range(len(candidates)), key=lambda index: len(candidates[index])):
        augment(expected_index, set())
    return matching


def compare(
    expected: ExpectedAction, actual: ExpectedAction | None, config: ToolCallComparatorConfig
) -> tuple[float, str]:
    """Reproduce the pinned source comparator, including its lenient default."""
    if actual is None:
        return 0.0, "no_action_found"
    if isinstance(expected, ExpectedMessage):
        # The pinned source scores any chat message as correct; it does not
        # compare message text. This is a source limitation, not a normalized
        # semantic equivalence rule.
        return (
            (1.0, "expected_chat_message_found")
            if isinstance(actual, ExpectedMessage)
            else (0.0, "no_expected_chat_message")
        )
    if isinstance(actual, ExpectedMessage):
        return 0.0, "no_expected_tool_call"
    expected_calls, actual_calls = _calls(expected), _calls(actual)
    if len(expected_calls) == len(actual_calls) == 1:
        return _tool_call_matches(expected_calls[0], actual_calls[0], config)
    if len(expected_calls) != len(actual_calls) and config.parallel_tool_call_rewarding:
        if len(actual_calls) < len(expected_calls) and not config.allow_subset:
            return 0.0, "function_call_batch_length_different"
        if len(actual_calls) > len(expected_calls) and not config.allow_superset:
            return 0.0, "function_call_batch_length_different"
    candidates: list[list[int]] = []
    failures: list[str] = []
    for expected_call in expected_calls:
        matches: list[int] = []
        category = "unexpected_tool"
        for index, actual_call in enumerate(actual_calls):
            matched, candidate_category = _tool_call_matches(expected_call, actual_call, config)
            if matched:
                matches.append(index)
            elif category == "unexpected_tool":
                category = candidate_category
        candidates.append(matches)
        failures.append(category)
    matched = _maximum_matching(candidates)
    matched_count = len(matched)
    if not config.parallel_tool_call_rewarding:
        reward = 1.0 if matched_count == len(expected_calls) else 0.0
    elif config.parallel_tool_call_reward_mode == ParallelToolCallRewardMode.F1:
        reward = 2 * matched_count / (len(expected_calls) + len(actual_calls))
    elif config.parallel_tool_call_reward_mode == ParallelToolCallRewardMode.FRACTIONAL:
        required = min(len(expected_calls), len(actual_calls)) if config.allow_subset else len(expected_calls)
        reward = matched_count / required if required else 0.0
    else:
        required = min(len(expected_calls), len(actual_calls)) if config.allow_subset else len(expected_calls)
        reward = 1.0 if matched_count == required else 0.0
    if reward == 1.0:
        return reward, "expected_tool_call" if len(expected_calls) == 1 else "expected_tool_call_batch"
    unmatched = [index for index in range(len(expected_calls)) if index not in set(matched.values())]
    if not unmatched or matched_count == len(actual_calls):
        return reward, "function_call_batch_length_different"
    return reward, failures[unmatched[0]]
