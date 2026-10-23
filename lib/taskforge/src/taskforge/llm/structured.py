# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed structured output through a JSON-schema ``response_format``, with a repair loop.

``complete_structured`` sends the request with ``response_format(output_type)``, the JSON schema
of the pydantic output type, and validates the reply against that type, running its validators
too. The chat template renders nothing from ``response_format``, so the loop appends
``schema_instruction(output_type)`` as the final user turn of every request it sends. The grammar
enforces the reply's shape, and the appended turn tells the language model what each field holds.

When the output limit cuts a reply off (``FinishReason.LENGTH``), the call fails at once with the
completions so far. For any other invalid reply the loop sends up to ``max_repairs`` repair requests,
each replaying the prior output and the validation error.

The loop calls its client through ``SchemaCompletions``, a protocol for sending a request
constrained to a schema, so any transport can serve it.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from taskforge.llm.client import Completion, FinishReason, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message

ERROR_TEXT_LIMIT = 4000

SCHEMA_INSTRUCTION_HEADER = "Reply with one JSON object with these fields:"
NESTED_FIELD_INDENT = "  "

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


def _definition(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve a ``$ref`` against the schema's ``$defs``, returning other schemas as they are."""
    if "$ref" in schema:
        return defs[schema["$ref"].rsplit("/", 1)[-1]]
    return schema


def _type_name(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> str:
    schema = _definition(schema, defs)
    if "anyOf" in schema:
        return " or ".join(_type_name(member, defs) for member in schema["anyOf"])
    if "enum" in schema:
        return "one of " + ", ".join(repr(value) for value in schema["enum"])
    if schema.get("type") == "array":
        return f"list of {_type_name(schema.get('items', {}), defs)}"
    return schema.get("type", "any")


def _nested_model(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Find the object a field holds, directly, as list items or as a member of an ``anyOf``."""
    schema = _definition(schema, defs)
    if "properties" in schema:
        return schema
    members = [schema["items"]] if "items" in schema else schema.get("anyOf", [])
    for member in members:
        if (nested := _nested_model(member, defs)) is not None:
            return nested
    return None


def _field_lines(model: Mapping[str, Any], defs: Mapping[str, Any], indent: str) -> list[str]:
    """Render a line per field of ``model``, then the fields of a nested object indented below it."""
    required = set(model.get("required", ()))
    lines = []
    for name, field in model["properties"].items():
        status = "required" if name in required else "optional"
        text = field.get("description") or field.get("title")
        lines.append(f"{indent}{name} ({_type_name(field, defs)}, {status})" + (f": {text}" if text else ""))
        nested = _nested_model(field, defs)
        if nested is not None and not indent:
            lines += _field_lines(nested, defs, NESTED_FIELD_INDENT)
    return lines


def schema_instruction(output_type: type[BaseModel]) -> str:
    """Render ``output_type``'s fields as the plain-text instruction for the final user turn.

    The text opens with ``SCHEMA_INSTRUCTION_HEADER`` and the model's docstring, then gives a line per
    field as ``<name> (<type>, required|optional): <description>``. A field without a description falls
    back to its title. Fields of a nested model go indented below its field, one level deep.
    """
    schema = output_type.model_json_schema()
    lead = [schema["description"]] if "description" in schema else []
    return "\n".join([SCHEMA_INSTRUCTION_HEADER, *lead, *_field_lines(schema, schema.get("$defs", {}), "")])


class CutReplyError(ValueError):
    """The output limit cut the reply off before it was complete."""


def _parsed[T: BaseModel](output_type: type[T], completion: Completion) -> T:
    """Validate the content as ``output_type``, raising ``ValueError`` for an invalid or cut-off reply."""
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

    A reply cut off by the output limit fails the call at once. Each repair request holds the original
    ``messages``, the latest invalid output and its validation error. Every request ends with a user
    turn holding ``schema_instruction(output_type)``: the grammar enforces the shape, and that turn
    tells the model what each field holds.

    Raises:
        StructuredOutputError: The output limit cut a reply off, or the last allowed reply was still
            invalid. The error holds every completion.
    """
    instruction: Message = {"role": "user", "content": schema_instruction(output_type)}
    completions = [await client.complete_schema([*messages, instruction], output_type, policy)]
    while True:
        try:
            return StructuredResult(_parsed(output_type, completions[-1]), tuple(completions))
        except CutReplyError as error:
            raise StructuredOutputError(error, completions) from error
        except ValueError as error:
            if len(completions) > max_repairs:
                raise StructuredOutputError(error, completions) from error
            repair = repair_messages(messages, completions[-1], error)
            completions.append(await client.complete_schema([*repair, instruction], output_type, policy))
