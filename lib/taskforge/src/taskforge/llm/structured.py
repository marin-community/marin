# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed structured output through a JSON-schema ``response_format``, with a repair loop.

A structured request carries ``response_format(output_type)``: the pydantic model's JSON schema,
which the server enforces by structured decoding, so the reply's content is one JSON object of
that shape. ``complete_structured`` validates the content with the model, which also runs its
validators. If the reply is invalid, a repair request follows that keeps the prior output in the
conversation as the assistant turn and states the validation error, up to ``max_repairs`` times.
A reply cut off by the output limit (``FinishReason.LENGTH``) is invalid even when it parses,
because a cut reply can be a valid object that is missing optional fields.

The loop is transport-free: its client is any ``SchemaCompletions``, which performs one request
constrained to a schema.
"""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from taskforge.llm.client import Completion, FinishReason, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message

ERROR_TEXT_LIMIT = 4000

REPAIR_PROMPT = (
    "Your previous output, shown above, failed validation:\n{error}\nReply with one complete, corrected JSON object."
)


class SchemaCompletions(Protocol):
    """A client that sends one request whose reply the server constrains to ``output_type``'s schema."""

    async def complete_schema(
        self, messages: Sequence[Message], output_type: type[BaseModel], policy: LLMPolicy
    ) -> Completion: ...


class StructuredOutputError(ValueError):
    """Neither the first reply nor any repair was valid."""

    def __init__(self, error: ValueError, completions: Sequence[Completion]):
        super().__init__(f"structured output invalid after {len(completions) - 1} repairs: {error}")
        self.completions = tuple(completions)


def response_format(output_type: type[BaseModel]) -> dict[str, object]:
    """The ``response_format`` request field that constrains a reply to ``output_type``'s JSON schema."""
    return {
        "type": "json_schema",
        "json_schema": {"name": output_type.__name__, "schema": output_type.model_json_schema(), "strict": True},
    }


def _parsed[T: BaseModel](output_type: type[T], completion: Completion) -> T:
    """Validate ``completion``'s content; raise ``ValueError`` (pydantic's ``ValidationError`` is one) if invalid."""
    if completion.finish_reason is FinishReason.LENGTH:
        raise ValueError("the reply was cut off by the output limit before it was complete")
    return output_type.model_validate_json(completion.content)


def repair_messages(messages: Sequence[Message], prior: Completion, error: ValueError) -> list[Message]:
    """Return ``messages`` plus the prior output as the assistant turn and the validation error."""
    prompt = REPAIR_PROMPT.format(error=str(error)[:ERROR_TEXT_LIMIT])
    return [*messages, {"role": "assistant", "content": prior.content}, {"role": "user", "content": prompt}]


async def complete_structured[T: BaseModel](
    client: SchemaCompletions,
    messages: Sequence[Message],
    output_type: type[T],
    policy: LLMPolicy,
    max_repairs: int,
) -> StructuredResult[T]:
    """Return the reply validated as ``output_type``, sending at most ``max_repairs`` repair requests.

    Each repair continues from the original ``messages`` with only the latest invalid output.

    Raises:
        StructuredOutputError: the last allowed reply was still invalid; it holds every completion.
    """
    completions = [await client.complete_schema(messages, output_type, policy)]
    while True:
        try:
            return StructuredResult(_parsed(output_type, completions[-1]), tuple(completions))
        except ValueError as error:
            if len(completions) > max_repairs:
                raise StructuredOutputError(error, completions) from error
            repair = repair_messages(messages, completions[-1], error)
            completions.append(await client.complete_schema(repair, output_type, policy))
