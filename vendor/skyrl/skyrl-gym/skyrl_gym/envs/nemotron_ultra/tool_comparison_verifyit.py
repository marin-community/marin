"""Score Nemotron tool arguments with verifyit's exact and numeric primitives."""

import json
import math
from typing import Any

from verifyit.adapters.skyrl import grade_literal_candidate
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.modes.grade_math import grade_numeric_candidate_float

from skyrl_gym.envs.nemotron_ultra.tool_call import StepRewardCategory

FLOAT_THRESHOLD = 1e-6
MESSAGE_SCHEMA = {
    "type": "object",
    "required": ["content", "tool_calls"],
    "properties": {
        "content": {"type": "string", "pattern": r"\S"},
        "tool_calls": {"type": "array", "maxItems": 0},
    },
}


def _validate_expected(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("expected tool arguments contain a nonfinite number")
    if isinstance(value, dict):
        for child in value.values():
            _validate_expected(child)
    elif isinstance(value, list):
        for child in value:
            _validate_expected(child)


def _compare(expected: Any, actual: Any) -> StepRewardCategory | None:
    if type(actual) is not type(expected):
        return StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT
        for key, value in expected.items():
            mismatch = _compare(value, actual[key])
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT
        for expected_item, actual_item in zip(expected, actual):
            mismatch = _compare(expected_item, actual_item)
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, float):
        if not math.isfinite(actual) or abs(actual - expected) >= FLOAT_THRESHOLD:
            return StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
        verdict = grade_numeric_candidate_float(expected, actual, tolerance_abs=FLOAT_THRESHOLD)
    else:
        verdict = grade_literal_candidate(json.dumps(expected), json.dumps(actual))
    return None if verdict.reward == 1.0 else StepRewardCategory.ARGUMENT_VALUE_DIFFERENT


def grade_expected_action_verifyit(
    expected_action: dict[str, Any], assistant_message: dict[str, Any]
) -> tuple[float, StepRewardCategory]:
    """Preserve the source action contract while grading leaves through verifyit."""
    if not isinstance(expected_action, dict):
        raise ValueError("expected action must be an object")
    expected_type = expected_action.get("type")
    if expected_type not in {"message", "function_call"}:
        raise ValueError(f"unsupported expected action type {expected_type!r}")
    tool_calls = assistant_message.get("tool_calls") or []
    content = assistant_message.get("content")
    if expected_type == "message":
        verdict = grade_json_schema_candidate(MESSAGE_SCHEMA, {"content": content, "tool_calls": tool_calls})
        if verdict.reward == 1.0:
            return 1.0, StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND
        return 0.0, StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE
    name = expected_action.get("name")
    argument_text = expected_action.get("arguments")
    if not isinstance(name, str) or not isinstance(argument_text, str):
        raise ValueError("expected tool call needs a name and JSON argument string")
    try:
        expected_arguments = json.loads(argument_text)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("expected tool arguments are not JSON") from error
    _validate_expected(expected_arguments)
    if not tool_calls:
        category = (
            StepRewardCategory.NO_EXPECTED_TOOL_CALL if isinstance(content, str) else StepRewardCategory.NO_ACTION_FOUND
        )
        return 0.0, category
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return 0.0, StepRewardCategory.NO_EXPECTED_TOOL_CALL
    call = tool_calls[0]
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        return 0.0, StepRewardCategory.NO_EXPECTED_TOOL_CALL
    actual = call["function"]
    actual_name = actual.get("name")
    if not isinstance(actual_name, str) or grade_literal_candidate(name, actual_name).reward != 1.0:
        return 0.0, StepRewardCategory.UNEXPECTED_TOOL
    try:
        actual_arguments = json.loads(actual.get("arguments"))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return 0.0, StepRewardCategory.ARGUMENTS_DECODE_ERROR
    mismatch = _compare(expected_arguments, actual_arguments)
    if mismatch is not None:
        return 0.0, mismatch
    return 1.0, StepRewardCategory.EXPECTED_TOOL_CALL
