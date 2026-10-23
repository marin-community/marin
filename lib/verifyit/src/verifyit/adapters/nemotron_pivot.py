# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Graders for NVIDIA's Nemotron agentic pivot releases, ported from NeMo Gym.

Each pivot is one expert turn of an agent trajectory. :func:`grade_tool_call` grades
``nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1`` with NeMo Gym's argument verifier
(``common/verification_utils.py``) at the SWE pivot agent's word-overlap threshold of 0.0.
:func:`grade_terminus` ports the string-only Terminus-2 verifier from
``resources_servers/terminus_judge``, used by ``nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1``.
Both keep NeMo's numeric typing. One deliberate difference: reasoning is stripped with
:func:`final_answer_text`, which also recognizes Grug's ``<|end_think|>``. Each verdict carries
named components in ``detail["components"]`` (each 0.0 or 1.0), so selection can switch criteria
later without regrading.
"""

import json
import operator
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any

from verifyit.grade import InvalidTask, Reward, scored
from verifyit.modes.grade_json_schema import grade_json_schema_candidate

TERMINUS_2_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis", "plan", "commands"],
    "properties": {
        "analysis": {"type": "string"},
        "plan": {"type": "string"},
        "task_complete": {"type": "boolean"},
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["keystrokes"],
                "properties": {"keystrokes": {"type": "string"}, "duration": {"type": "number"}},
            },
        },
    },
}


def final_answer_text(text: str) -> str:
    """Strip explicit reasoning blocks; an unfinished block has no final answer.

    A closing marker alone ends reasoning whose opening marker the chat template supplied.
    """
    delimiters = (("<think>", "</think>"), ("<thinking>", "</thinking>"), ("<|start_think|>", "<|end_think|>"))
    for opening, closing in delimiters:
        text = re.sub(re.escape(opening) + ".*?" + re.escape(closing), "", text, flags=re.DOTALL)
    last_closing = max((text.rfind(closing) + len(closing) for _, closing in delimiters if closing in text), default=0)
    text = text[last_closing:]
    if any(opening in text for opening, _ in delimiters):
        return ""
    return text.strip().removesuffix("<|eot_id|>").strip()


def arguments_match(expected: Any, actual: Any) -> bool:
    """Strict recursive argument comparison: every value equal, strings included.

    This is MarinSkyRL's ``grade_expected_action`` comparison. The type check is ``isinstance``,
    so a bool satisfies an expected int but an int does not satisfy an expected float. Floats
    match within 1e-6.
    """
    return _values_match(expected, actual, operator.eq)


def word_overlap_match(expected: Any, actual: Any) -> bool:
    """NeMo Gym's argument verifier at the SWE pivot agent's word-overlap threshold of 0.0.

    NeMo scores a pair of multi-word strings as shared words over the two strings' total words
    (identical strings score 0.5) against a threshold. At 0.0, any two strings of at least two
    words match, so only tool names, keys, list lengths, and single-word strings must agree.
    Like :func:`arguments_match` otherwise.
    """
    return _values_match(expected, actual, _words_overlap)


def _words_overlap(expected: str, actual: str) -> bool:
    return min(len(expected.split()), len(actual.split())) >= 2 or expected == actual


def _values_match(expected: Any, actual: Any, strings_match: Callable[[str, str], bool]) -> bool:
    if not isinstance(actual, type(expected)):
        return False
    match expected:
        case dict():
            return expected.keys() == actual.keys() and all(
                _values_match(value, actual[key], strings_match) for key, value in expected.items()
            )
        case list():
            return len(expected) == len(actual) and all(
                _values_match(left, right, strings_match) for left, right in zip(expected, actual, strict=True)
            )
        case float():
            return abs(actual - expected) < 1e-6
        case str():
            return strings_match(expected, actual)
        case _:
            return expected == actual


def grade_tool_call(expected_action: dict[str, Any], message: dict[str, Any]) -> Reward:
    """Score an OpenAI-style assistant message against a pivot's expected action.

    The reward is ``nemo``, NeMo Gym's SWE pivot verifier with its default of not rewarding
    parallel calls: a ``message`` action accepts any text reply without tool calls; a
    ``function_call`` action accepts a reply with a call to the expected function whose arguments
    pass :func:`word_overlap_match`, and further calls cost nothing. ``detail["components"]`` scores
    the reply under each criterion, each satisfied by any one call:

    - ``tool_name``: the right tool, or no tool for a message action. The PivotRL paper's SWE
      verifier, which "matches tool-call names only".
    - ``nemo``: NeMo Gym's SWE pivot verifier, equal to the reward.
    - ``exact``: every argument value equal (:func:`arguments_match`) and, for a message action,
      non-empty content after reasoning.
    """
    tool_calls = message.get("tool_calls") or []
    match expected_action.get("type"):
        case "message":
            return _message_verdict(message, tool_calls)
        case "function_call":
            return _function_call_verdict(expected_action, tool_calls)
        case unsupported:
            raise InvalidTask(f"unsupported expected action type {unsupported!r}")


def _message_verdict(message: dict[str, Any], tool_calls: list[dict[str, Any]]) -> Reward:
    content = message.get("content")
    answered = not tool_calls
    exact = answered and bool(final_answer_text(content or ""))
    nemo = answered and isinstance(content, str)
    return _tool_call_verdict("chat_message" if nemo else "no_chat_message", tool_name=answered, nemo=nemo, exact=exact)


def _function_call_verdict(expected_action: dict[str, Any], tool_calls: list[dict[str, Any]]) -> Reward:
    try:
        expected_arguments = json.loads(expected_action["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise InvalidTask("expected tool call needs JSON arguments") from error
    if not tool_calls:
        return _tool_call_verdict("no_tool_call")
    calls = [_call_match(expected_action.get("name"), expected_arguments, call) for call in tool_calls]
    tool_name, nemo, exact = (any(match[i] for match in calls) for i in range(3))
    if nemo:
        reason = "expected_tool_call"
    elif tool_name:
        reason = "arguments_differ"
    else:
        reason = "unexpected_tool"
    return _tool_call_verdict(reason, tool_name=tool_name, nemo=nemo, exact=exact, tool_calls=len(tool_calls))


def _call_match(name: str | None, expected_arguments: Any, call: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Whether one call has the expected name, passes NeMo's argument match, and matches exactly."""
    function = call.get("function") or {}
    if function.get("name") != name:
        return False, False, False
    try:
        arguments = json.loads(function.get("arguments"))
    except (TypeError, json.JSONDecodeError):
        return True, False, False
    return True, word_overlap_match(expected_arguments, arguments), arguments_match(expected_arguments, arguments)


def _tool_call_verdict(
    reason: str, *, tool_name: bool = False, nemo: bool = False, exact: bool = False, **detail: Any
) -> Reward:
    components = {"tool_name": float(tool_name), "nemo": float(nemo), "exact": float(exact)}
    return scored(float(nemo), reason=reason, components=components, **detail)


def grade_terminus(expected_answer: str, text: str, threshold: float | None = None) -> Reward:
    """Score a Terminus-2 JSON reply by keystroke similarity to the expected action.

    The reply is the text after any reasoning block (:func:`final_answer_text`), which handles
    both ``</think>`` and Grug's ``<|end_think|>``; NeMo Gym splits on ``</think>`` only. It must
    match the Terminus-2 schema and must not leave a completed task unfinished. The reward is
    ``string_90``: the concatenated keystrokes are at least ``threshold`` similar (default 0.9) to
    the expected ones. ``detail["components"]`` also records:

    - ``schema_completion``: a schema-valid reply that does not leave a completed task unfinished.
    - ``exact_commands``: the same keystrokes, command by command.
    - ``string_90``: equal to the reward.
    """
    expected = json.loads(expected_answer)
    if grade_json_schema_candidate(TERMINUS_2_SCHEMA, expected).reward != 1.0:
        raise InvalidTask("expected answer is not a Terminus-2 action")
    try:
        candidate = json.loads(final_answer_text(text))
    except json.JSONDecodeError:
        return _terminus_verdict("invalid_json")
    schema = grade_json_schema_candidate(TERMINUS_2_SCHEMA, candidate)
    if schema.reward != 1.0:
        return _terminus_verdict("schema_violation", error=schema.detail.get("error"))
    if expected.get("task_complete", False) and not candidate.get("task_complete", False):
        return _terminus_verdict("task_incomplete")

    expected_keystrokes = [command["keystrokes"] for command in expected["commands"]]
    keystrokes = [command["keystrokes"] for command in candidate["commands"]]
    if bool(expected_keystrokes) != bool(keystrokes):
        return _terminus_verdict("commands_presence_differs", schema_completion=True, similarity=0.0)
    similarity = SequenceMatcher(None, "".join(expected_keystrokes), "".join(keystrokes)).ratio()
    similar = similarity >= (0.9 if threshold is None else threshold)
    return _terminus_verdict(
        "similar" if similar else "dissimilar",
        schema_completion=True,
        exact_commands=keystrokes == expected_keystrokes,
        similar=similar,
        similarity=similarity,
    )


def _terminus_verdict(
    reason: str, *, schema_completion: bool = False, exact_commands: bool = False, similar: bool = False, **detail: Any
) -> Reward:
    components = {
        "schema_completion": float(schema_completion),
        "exact_commands": float(exact_commands),
        "string_90": float(similar),
    }
    return scored(float(similar), reason=reason, components=components, **detail)
