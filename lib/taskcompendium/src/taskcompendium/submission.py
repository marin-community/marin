# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.final_action import SubmittedCalls, decode_action, parse_arguments, unique_json_fields
from taskcompendium.models import AnswerType, TaskSpec, format_conversation

ANSWER_CALL_NAME = "submit_answer"
ANSWER_FIELD = "answer"


def answer_call_tool() -> dict[str, object]:
    """Return the function definition advertised by the answer-call convention."""
    return {
        "type": "function",
        "function": {
            "name": ANSWER_CALL_NAME,
            "description": "Submit the final answer to the task.",
            "parameters": {
                "type": "object",
                "properties": {ANSWER_FIELD: {"type": "string"}},
                "required": [ANSWER_FIELD],
                "additionalProperties": False,
            },
        },
    }


class AnswerFormat(StrEnum):
    """The envelope used to deliver a result."""

    PLAIN = "plain"
    JSON = "json"
    ANSWER_CALL = "answer_call"
    FINAL_ACTION = "final_action"


class SubmissionConvention(BaseModel):
    """How a result is requested, delivered, and extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    answer_format: AnswerFormat

    @model_validator(mode="after")
    def validate_convention(self) -> "SubmissionConvention":
        if not self.id:
            raise ValueError("A submission convention id is required")
        return self

    def supports(self, answer_type: AnswerType) -> bool:
        """Whether this convention can carry the semantic result."""
        if self.answer_format == AnswerFormat.FINAL_ACTION:
            return answer_type == AnswerType.NATIVE_ACTION
        return answer_type in (AnswerType.TEXT, AnswerType.NUMBER)


def submission_compatible(specification: TaskSpec, convention: SubmissionConvention) -> bool:
    if not convention.supports(specification.answer_type):
        return False
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return bool(specification.tools.functions)
    return (
        not specification.tools.functions
        and specification.tools.tool_choice is None
        and specification.tools.parallel_tool_calls is None
    )


def submission_instruction(convention: SubmissionConvention) -> str:
    """Return the instruction added after a conversation prefix."""
    if convention.answer_format == AnswerFormat.PLAIN:
        return "Give your answer as plain text."
    if convention.answer_format == AnswerFormat.JSON:
        return f'Give your answer as a JSON object with an "{ANSWER_FIELD}" field.'
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        return f'Call {ANSWER_CALL_NAME} with your final answer as the "{ANSWER_FIELD}" string.'
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return ""
    raise ValueError(f"Unsupported answer format: {convention.answer_format}")


def render_instruction(specification: TaskSpec, convention: SubmissionConvention) -> str:
    """Return Harbor instruction text for the selected convention."""
    context = specification.context
    if not submission_compatible(specification, convention):
        raise ValueError(
            f"Submission convention {convention.id!r} cannot carry {specification.answer_type.value!r} in this context"
        )
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return format_conversation(context.events)
    return f"{format_conversation(context.events)}\n\n{submission_instruction(convention)}\n"


def extract_answer(response: str | None, convention: SubmissionConvention) -> str:
    """Decode the selected submission convention without guessing a format."""
    if response is None or not response.strip():
        raise ValueError("Final answer is empty")
    if convention.answer_format == AnswerFormat.PLAIN:
        return response
    if convention.answer_format == AnswerFormat.JSON:
        value = json.loads(response, object_pairs_hook=unique_json_fields)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get(ANSWER_FIELD), str)
            or not value[ANSWER_FIELD].strip()
        ):
            raise ValueError("JSON submission requires a nonempty string answer")
        return value[ANSWER_FIELD]
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        message = json.loads(response, object_pairs_hook=unique_json_fields)
        if not isinstance(message, dict):
            raise ValueError("Answer call requires an assistant message object")
        action = decode_action(message)
        if not isinstance(action, SubmittedCalls) or len(action.calls) != 1 or action.calls[0].name != ANSWER_CALL_NAME:
            raise ValueError(f"Answer call requires one {ANSWER_CALL_NAME} function call")
        arguments = parse_arguments(action.calls[0].arguments)
        if (
            set(arguments) != {ANSWER_FIELD}
            or not isinstance(arguments[ANSWER_FIELD], str)
            or not arguments[ANSWER_FIELD].strip()
        ):
            raise ValueError("Answer call requires a nonempty string answer")
        return arguments[ANSWER_FIELD]
    raise ValueError(f"Unsupported answer format: {convention.answer_format}")
