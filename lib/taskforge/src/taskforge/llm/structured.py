# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed structured output through a JSON-schema ``response_format``, with a repair loop.

``complete_structured`` sends the request with ``response_format(output_type)``, the pydantic
model's JSON schema. vLLM decodes the reply under that schema, so its content is a single JSON
object matching it. The loop validates the content with the pydantic model, running its validators
too. For an invalid reply it sends a repair request, with the prior output as the assistant turn and
the validation error in a user turn, up to ``max_repairs`` times.

When the output limit cuts a reply off (``FinishReason.LENGTH``), the loop raises at once, even when
the content parses, since a cut reply can be a valid object missing optional fields. A repair would
replay the cut reply as the assistant turn, so its prompt could overflow the context window, and the
repaired reply would face the same output limit.

The loop calls its client through ``SchemaCompletions``, a protocol for sending a request
constrained to a schema, so any transport can serve it.
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
    """Protocol for sending a request with its reply constrained to ``output_type``'s JSON schema."""

    async def complete_schema(
        self, messages: Sequence[Message], output_type: type[BaseModel], policy: LLMPolicy
    ) -> Completion: ...


class StructuredOutputError(ValueError):
    """The output limit cut a reply off, or the first reply and every repair failed validation."""

    def __init__(self, error: ValueError, completions: Sequence[Completion]):
        super().__init__(f"structured output invalid after {len(completions) - 1} repairs: {error}")
        self.completions = tuple(completions)


def response_format(output_type: type[BaseModel]) -> dict[str, object]:
    """Build the ``response_format`` request field for ``output_type``'s JSON schema."""
    return {
        "type": "json_schema",
        "json_schema": {"name": output_type.__name__, "schema": output_type.model_json_schema(), "strict": True},
    }


class CutReplyError(ValueError):
    """The output limit cut the reply off before it was complete."""


def _parsed[T: BaseModel](output_type: type[T], completion: Completion) -> T:
    """Validate the content as ``output_type``, raising ``ValueError`` for an invalid reply.

    Pydantic's ``ValidationError`` is a ``ValueError``. For a reply cut off by the output limit it raises
    ``CutReplyError``, so the caller can fail the call without a repair.
    """
    if completion.finish_reason is FinishReason.LENGTH:
        raise CutReplyError("the reply was cut off by the output limit before it was complete")
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

    Each repair request starts from the original ``messages`` and carries only the latest invalid output.
    When the output limit cuts a reply off, the loop raises at once, since a repair would replay the cut
    reply and could overflow the context window.

    Raises:
        StructuredOutputError: The output limit cut a reply off, or the last allowed reply was still
            invalid. The error holds every completion.
    """
    completions = [await client.complete_schema(messages, output_type, policy)]
    while True:
        try:
            return StructuredResult(_parsed(output_type, completions[-1]), tuple(completions))
        except CutReplyError as error:
            raise StructuredOutputError(error, completions) from error
        except ValueError as error:
            if len(completions) > max_repairs:
                raise StructuredOutputError(error, completions) from error
            repair = repair_messages(messages, completions[-1], error)
            completions.append(await client.complete_schema(repair, output_type, policy))
