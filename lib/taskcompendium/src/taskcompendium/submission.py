# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.models import AnswerType, TaskSpec


class AnswerFormat(StrEnum):
    """A model-visible envelope for a submitted answer."""

    PLAIN = "plain"
    JSON = "json"
    FILE = "file"
    WORKSPACE = "workspace"


class SubmissionConvention(BaseModel):
    """How a result is requested, delivered, and extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    answer_format: AnswerFormat
    output_path: str | None = None

    @model_validator(mode="after")
    def validate_convention(self) -> "SubmissionConvention":
        if not self.id:
            raise ValueError("A submission convention id is required")
        if self.answer_format == AnswerFormat.FILE:
            path = self.output_path
            if (
                path is None
                or path.startswith("/")
                or "\\" in path
                or ":" in path.split("/")[0]
                or any(ord(character) < 32 for character in path)
                or any(part in ("", ".", "..") for part in path.split("/"))
            ):
                raise ValueError("A file submission requires a normalized relative output path")
            if PurePosixPath(path).as_posix() != path:
                raise ValueError("A file submission requires a normalized relative output path")
        elif self.output_path is not None:
            raise ValueError("Only a file submission may specify an output path")
        return self

    def supports(self, answer_type: AnswerType) -> bool:
        if self.answer_format in (AnswerFormat.PLAIN, AnswerFormat.JSON):
            return answer_type in (AnswerType.TEXT, AnswerType.NUMBER)
        if self.answer_format == AnswerFormat.FILE:
            return answer_type == AnswerType.FILE
        if self.answer_format == AnswerFormat.WORKSPACE:
            return answer_type == AnswerType.WORKSPACE_STATE
        raise ValueError(f"Unsupported answer format: {self.answer_format}")


def _object_with_unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate fields."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def render_instruction(specification: TaskSpec, convention: SubmissionConvention) -> str:
    """Return the public request with its submission instructions attached."""
    if not convention.supports(specification.answer_type):
        raise ValueError(f"Submission convention {convention.id!r} cannot carry {specification.answer_type.value!r}")
    if convention.answer_format == AnswerFormat.PLAIN:
        suffix = "Give your answer as plain text."
    elif convention.answer_format == AnswerFormat.JSON:
        suffix = 'Give your answer as a JSON object with an "answer" field.'
    elif convention.answer_format == AnswerFormat.FILE:
        suffix = f"Write your final file to /app/{convention.output_path}."
    elif convention.answer_format == AnswerFormat.WORKSPACE:
        suffix = "Complete the requested changes in /app."
    else:
        raise ValueError(f"Unsupported answer format: {convention.answer_format}")
    return f"{specification.instructions.rstrip()}\n\n{suffix}\n"


def extract_answer(response: str | None, convention: SubmissionConvention) -> str:
    """Decode the selected submission convention without guessing a format."""
    if response is None or not response.strip():
        raise ValueError("Final answer is empty")
    if convention.answer_format == AnswerFormat.PLAIN:
        return response
    elif convention.answer_format == AnswerFormat.JSON:
        value = json.loads(response, object_pairs_hook=_object_with_unique_fields)
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
            raise ValueError("JSON submission requires a nonempty string answer")
        return value["answer"]
    else:
        raise ValueError(f"Unsupported answer format: {convention.answer_format}")
