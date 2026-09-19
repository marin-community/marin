# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-visible renderings of a single semantic answer task."""

import json
from dataclasses import dataclass
from enum import StrEnum

from taskcompendium.models import AnswerKind, TaskSpec


class AnswerFormat(StrEnum):
    PLAIN = "plain"
    JSON = "json"


@dataclass(frozen=True)
class Rendering:
    id: str
    answer_format: AnswerFormat


def render_instruction(specification: TaskSpec, rendering: Rendering) -> str:
    """Return the public request with only its answer convention attached."""
    if specification.answer_kind is AnswerKind.OPTION_LETTER:
        answer = "the selected option letter"
    elif specification.answer_kind is AnswerKind.TEXT:
        answer = "your answer"
    else:
        raise ValueError(f"Unsupported answer kind: {specification.answer_kind}")
    if rendering.answer_format == AnswerFormat.PLAIN:
        suffix = f"Return {answer} as plain text."
    elif rendering.answer_format == AnswerFormat.JSON:
        suffix = f'Return a JSON object with an "answer" field containing {answer}.'
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
