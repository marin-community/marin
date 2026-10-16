# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Modes pivot-tool-call and terminus: one expert turn of NVIDIA's Nemotron agentic pivot releases.

Both apply :mod:`verifyit.adapters.nemotron_pivot` without a judge, so the reward is deterministic:
``pivot-tool-call`` is NeMo Gym's SWE pivot argument comparison over the whole assistant message,
and ``terminus`` is Terminus-2 keystroke similarity over the reply text.
"""

import json
from pathlib import Path
from typing import Any

from verifyit.adapters.nemotron_pivot import TERMINUS_2_SCHEMA, grade_terminus, grade_tool_call
from verifyit.grade import InvalidTask, Reward, read_output, scored
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.spec import PivotToolCallSpec, TerminusSpec


def _json_object(text: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidTask(f"{name} must be a JSON object") from error
    if not isinstance(value, dict):
        raise InvalidTask(f"{name} must be a JSON object")
    return value


def validate_pivot_tool_call(spec: PivotToolCallSpec) -> None:
    action = _json_object(spec.expected_action, "expected_action")
    match action.get("type"):
        case "message":
            return
        case "function_call":
            if not isinstance(action.get("name"), str) or not action["name"]:
                raise InvalidTask("an expected function call needs a name")
            _json_object(action.get("arguments") or "", "expected function call arguments")
        case unsupported:
            raise InvalidTask(f"unsupported expected action type {unsupported!r}")


def grade_pivot_tool_call(spec: PivotToolCallSpec, message: dict[str, Any]) -> Reward:
    return grade_tool_call(json.loads(spec.expected_action), message)


def validate_terminus(spec: TerminusSpec) -> None:
    expected = _json_object(spec.expected_answer, "expected_answer")
    if grade_json_schema_candidate(TERMINUS_2_SCHEMA, expected).reward != 1.0:
        raise InvalidTask("expected answer is not a Terminus-2 action")
    if spec.threshold is not None and not 0 < spec.threshold <= 1:
        raise InvalidTask("threshold must be in (0, 1]")


def grade_terminus_candidate(spec: TerminusSpec, text: str) -> Reward:
    return grade_terminus(spec.expected_answer, text, spec.threshold)


def output_message(text: str) -> dict[str, Any] | None:
    """The assistant message an answer file holds as JSON, or ``None`` when it holds none."""
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        return None
    return message if isinstance(message, dict) else None


def grade(spec: PivotToolCallSpec | TerminusSpec, _tests_dir: Path, workspace: Path) -> Reward:
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    if isinstance(spec, TerminusSpec):
        validate_terminus(spec)
        return grade_terminus_candidate(spec, text)
    validate_pivot_tool_call(spec)
    message = output_message(text)
    if message is None:
        return scored(0.0, reason="invalid_message")
    return grade_pivot_tool_call(spec, message)
