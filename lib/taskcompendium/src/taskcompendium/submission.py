# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.final_action import SubmittedCalls, decode_action, parse_arguments, unique_json_fields
from taskcompendium.models import AnswerType, TaskSpec, format_native_messages

ANSWER_CALL_NAME = "submit_answer"
ANSWER_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": ANSWER_CALL_NAME,
        "description": "Submit the final answer to the task.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
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
        request = specification.native_action_request
        return request is not None and specification.instructions == format_native_messages(request.messages)
    return specification.native_action_request is None


def render_instruction(specification: TaskSpec, convention: SubmissionConvention) -> str:
    """Return Harbor instruction text for the selected convention."""
    if not convention.supports(specification.answer_type):
        raise ValueError(f"Submission convention {convention.id!r} cannot carry {specification.answer_type.value!r}")
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        request = specification.native_action_request
        if request is None:
            raise ValueError("Final-action task requires a source request")
        if specification.instructions != format_native_messages(request.messages):
            raise ValueError("Final-action instructions differ from source messages")
        return specification.instructions
    if specification.native_action_request is not None:
        raise ValueError("Text tasks cannot carry native action requests")
    if convention.answer_format == AnswerFormat.PLAIN:
        suffix = "Give your answer as plain text."
    elif convention.answer_format == AnswerFormat.JSON:
        suffix = 'Give your answer as a JSON object with an "answer" field.'
    elif convention.answer_format == AnswerFormat.ANSWER_CALL:
        suffix = f'Call {ANSWER_CALL_NAME} with your final answer as the "answer" string.'
    else:
        raise ValueError(f"Unsupported answer format: {convention.answer_format}")
    return f"{specification.instructions.rstrip()}\n\n{suffix}\n"


def extract_answer(response: str | None, convention: SubmissionConvention) -> str:
    """Decode the selected submission convention without guessing a format."""
    if response is None or not response.strip():
        raise ValueError("Final answer is empty")
    if convention.answer_format == AnswerFormat.PLAIN:
        return response
    if convention.answer_format == AnswerFormat.JSON:
        value = json.loads(response, object_pairs_hook=unique_json_fields)
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
            raise ValueError("JSON submission requires a nonempty string answer")
        return value["answer"]
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        message = json.loads(response, object_pairs_hook=unique_json_fields)
        if not isinstance(message, dict):
            raise ValueError("Answer call requires an assistant message object")
        action = decode_action(message)
        if not isinstance(action, SubmittedCalls) or len(action.calls) != 1 or action.calls[0].name != ANSWER_CALL_NAME:
            raise ValueError(f"Answer call requires one {ANSWER_CALL_NAME} function call")
        arguments = parse_arguments(action.calls[0].arguments)
        if set(arguments) != {"answer"} or not isinstance(arguments["answer"], str) or not arguments["answer"].strip():
            raise ValueError("Answer call requires a nonempty string answer")
        return arguments["answer"]
    raise ValueError(f"Unsupported answer format: {convention.answer_format}")
