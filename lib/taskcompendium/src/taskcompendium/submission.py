# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.models import AnswerType, NativeFunction, TaskSpec


class AnswerFormat(StrEnum):
    """The envelope used to deliver a result."""

    PLAIN = "plain"
    JSON = "json"
    FINAL_ACTION = "final_action"


class NativeMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    content: str

    @model_validator(mode="after")
    def validate_message(self) -> "NativeMessage":
        if self.role not in {"system", "user", "assistant"} or not self.content.strip():
            raise ValueError("Native messages require a supported role and nonempty content")
        return self


def format_native_messages(messages: tuple[NativeMessage, ...]) -> str:
    """Produce the Harbor instruction view of structured source messages."""
    return "\n\n".join(f"{message.role.title()}:\n{message.content.strip()}" for message in messages)


class SubmissionConvention(BaseModel):
    """How a result is requested, delivered, and extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    answer_format: AnswerFormat
    functions: tuple[NativeFunction, ...] = ()
    messages: tuple[NativeMessage, ...] = ()
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None

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


def _object_with_unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate fields."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def submission_compatible(specification: TaskSpec, convention: SubmissionConvention) -> bool:
    if not convention.supports(specification.answer_type):
        return False
    if (
        specification.permitted_submission_conventions is not None
        and convention.id not in specification.permitted_submission_conventions
    ):
        return False
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return bool(
            convention.functions and convention.messages
        ) and specification.instructions == format_native_messages(convention.messages)
    return not (
        convention.functions
        or convention.messages
        or convention.tool_choice is not None
        or convention.parallel_tool_calls is not None
    )


def render_instruction(specification: TaskSpec, convention: SubmissionConvention) -> str:
    """Return Harbor instruction text for the selected convention."""
    if not convention.supports(specification.answer_type):
        raise ValueError(f"Submission convention {convention.id!r} cannot carry {specification.answer_type.value!r}")
    if (
        specification.permitted_submission_conventions is not None
        and convention.id not in specification.permitted_submission_conventions
    ):
        raise ValueError(f"Task {specification.id!r} does not permit submission convention {convention.id!r}")
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        if not convention.functions or not convention.messages:
            raise ValueError("Final-action convention requires advertised functions and source messages")
        if specification.instructions != format_native_messages(convention.messages):
            raise ValueError("Final-action instructions differ from source messages")
        return specification.instructions
    if (
        convention.functions
        or convention.messages
        or convention.tool_choice is not None
        or convention.parallel_tool_calls is not None
    ):
        raise ValueError("Text conventions cannot carry native action requests")
    if convention.answer_format == AnswerFormat.PLAIN:
        suffix = "Give your answer as plain text."
    elif convention.answer_format == AnswerFormat.JSON:
        suffix = 'Give your answer as a JSON object with an "answer" field.'
    else:
        raise ValueError(f"Unsupported answer format: {convention.answer_format}")
    return f"{specification.instructions.rstrip()}\n\n{suffix}\n"


def extract_answer(response: str | None, convention: SubmissionConvention) -> str:
    """Decode the selected submission convention without guessing a format."""
    if response is None or not response.strip():
        raise ValueError("Final answer is empty")
    if convention.answer_format == AnswerFormat.PLAIN:
        return response
    if convention.answer_format != AnswerFormat.JSON:
        raise ValueError(f"Unsupported answer format: {convention.answer_format}")

    value = json.loads(response, object_pairs_hook=_object_with_unique_fields)
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise ValueError("JSON submission requires a nonempty string answer")
    return value["answer"]
