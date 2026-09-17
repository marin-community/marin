# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-visible renderings of a single semantic answer task."""

import json
from dataclasses import dataclass
from enum import StrEnum

from taskcompendium.models import NativeFunction, TaskSpec


class AnswerFormat(StrEnum):
    PLAIN = "plain"
    JSON = "json"
    FINAL_ACTION = "final_action"


@dataclass(frozen=True)
class NativeMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"} or not self.content.strip():
            raise ValueError("Native messages require a supported role and nonempty content")


def format_native_messages(messages: tuple[NativeMessage, ...]) -> str:
    """Produce the Harbor instruction view of structured source messages."""
    return "\n\n".join(f"{message.role.title()}:\n{message.content.strip()}" for message in messages)


@dataclass(frozen=True)
class Rendering:
    id: str
    answer_format: AnswerFormat
    functions: tuple[NativeFunction, ...] = ()
    messages: tuple[NativeMessage, ...] = ()
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None


def render_instruction(specification: TaskSpec, rendering: Rendering) -> str:
    """Return the public request with only its answer convention attached."""
    if rendering.answer_format == AnswerFormat.FINAL_ACTION:
        if not rendering.functions or not rendering.messages:
            raise ValueError("Final-action rendering requires advertised functions and source messages")
        if specification.instructions != format_native_messages(rendering.messages):
            raise ValueError("Final-action instructions differ from source messages")
        return specification.instructions
    if (
        rendering.functions
        or rendering.messages
        or rendering.tool_choice is not None
        or rendering.parallel_tool_calls is not None
    ):
        raise ValueError("Answer renderings cannot carry native action requests")
    if rendering.answer_format == AnswerFormat.PLAIN:
        suffix = "Give your answer as plain text."
    elif rendering.answer_format == AnswerFormat.JSON:
        suffix = 'Give your answer as a JSON object with an "answer" field.'
    else:
        raise ValueError(f"Unsupported answer format: {rendering.answer_format}")
    return f"{specification.instructions.rstrip()}\n\n{suffix}\n"


def extract_answer(response: str | None, rendering: Rendering) -> str:
    """Decode the selected submission convention without guessing a format."""
    if response is None or not response.strip():
        raise ValueError("Final answer is empty")
    if rendering.answer_format == AnswerFormat.PLAIN:
        return response
    if rendering.answer_format != AnswerFormat.JSON:
        raise ValueError(f"Unsupported answer format: {rendering.answer_format}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    value = json.loads(response, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise ValueError("JSON submission requires a nonempty string answer")
    return value["answer"]
