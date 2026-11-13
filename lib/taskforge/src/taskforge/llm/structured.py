# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed structured output through a JSON-schema ``response_format``, with a repair loop.

``complete_structured`` sends the request with ``response_format(output_type)``, the JSON schema
of the pydantic output type, and validates the reply against that type, running its validators
too. The chat template renders nothing from ``response_format``, so the loop appends
``schema_instruction(output_type)`` as the final user turn of every request it sends. The server's
grammar enforces the schema, including bounds such as ``maxLength``. The appended turn states each
field's type, bounds and meaning in words, so the model can keep each value within its bounds as it
drafts. When a draft runs past ``maxLength``, the grammar closes the string mid-word.

When the output limit cuts a reply off (``FinishReason.LENGTH``), the call fails at once with the
completions so far. For any other invalid reply the loop sends up to ``max_repairs`` repair requests,
each replaying the prior output and the validation error.

The loop calls its client through ``SchemaCompletions``, a protocol for sending a request
constrained to a schema, so any transport can serve it.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from taskforge.llm.client import Completion, FinishReason, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message

ERROR_TEXT_LIMIT = 4000

SCHEMA_INSTRUCTION_HEADER = "Reply with one JSON object with these fields:"
NESTED_FIELD_INDENT = "  "
REPEATED_MODEL_LINE = "{name} (same fields as above)"

COUNT_BOUNDS = {
    "minLength": ("at least", "character"),
    "maxLength": ("at most", "character"),
    "minItems": ("at least", "item"),
    "maxItems": ("at most", "item"),
}
NUMBER_BOUNDS = {
    "minimum": "at least",
    "maximum": "at most",
    "exclusiveMinimum": "greater than",
    "exclusiveMaximum": "less than",
}

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
        return " or ".join(dict.fromkeys(_type_name(member, defs) for member in schema["anyOf"]))
    if schema.get("type") == "array":
        return f"list of {_type_name(schema.get('items', {}), defs)}"
    return schema.get("type", "any")


def _keyword_text(schema: Mapping[str, Any]) -> list[str]:
    """Put a schema's own value constraints into words, leaving its ``anyOf`` members and items aside."""
    words = []
    if "const" in schema:
        words.append(f"exactly {json.dumps(schema['const'])}")
    if "enum" in schema:
        words.append("one of: " + ", ".join(json.dumps(value) for value in schema["enum"]))
    for keyword, (relation, unit) in COUNT_BOUNDS.items():
        if keyword in schema:
            count = schema[keyword]
            words.append(f"{relation} {count} {unit}" + ("" if count == 1 else "s"))
    for keyword, relation in NUMBER_BOUNDS.items():
        if keyword in schema:
            words.append(f"{relation} {schema[keyword]}")
    if "pattern" in schema:
        words.append(f"matching {schema['pattern']}")
    return words


def _constraints(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> list[str]:
    """Describe a schema's constraints, keeping the alternatives of an ``anyOf`` apart.

    Constraints on list items get an ``each`` prefix, as in ``each at most 200 characters``. When one
    ``anyOf`` member alone carries constraints, as for an optional field, they render as they would
    on a plain field. When several members carry constraints, the members become one term of
    alternatives with each member's constraints joined by ``and``, as in
    ``at least 0 and at most 10 or at least 20 and at most 30``.
    """
    schema = _definition(schema, defs)
    if "anyOf" in schema:
        members = [words for member in schema["anyOf"] if (words := _constraints(member, defs))]
        if len(members) > 1:
            return [" or ".join(" and ".join(words) for words in members)]
        return members[0] if members else []
    words = _keyword_text(schema)
    if "items" in schema:
        words += [f"each {text}" for text in _constraints(schema["items"], defs)]
    return words


def _ref_name(schema: Mapping[str, Any]) -> str | None:
    """Return the ``$defs`` key a schema's ``$ref`` points at, or ``None`` for a schema without one."""
    return schema["$ref"].rsplit("/", 1)[-1] if "$ref" in schema else None


def _nested_model(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any]] | None:
    """Find the object a field holds, directly, as list items or as a member of an ``anyOf``.

    The object comes back with its ``$defs`` key, or with ``None`` in place of the key for an object
    written inline.
    """
    resolved = _definition(schema, defs)
    if "properties" in resolved:
        return _ref_name(schema), resolved
    members = [resolved["items"]] if "items" in resolved else resolved.get("anyOf", [])
    for member in members:
        if (nested := _nested_model(member, defs)) is not None:
            return nested
    return None


def _lead_lines(model: Mapping[str, Any], indent: str) -> list[str]:
    """Indent each line of a model's docstring, giving an empty list for a model without one."""
    return [indent + line if line else line for line in model.get("description", "").splitlines()]


def _field_lines(model: Mapping[str, Any], defs: Mapping[str, Any], indent: str, path: frozenset[str]) -> list[str]:
    """Render a line per field of ``model``, each followed by the object it holds, one indent further in.

    For a nested object the function writes the docstring and then recurses into the object's fields.
    ``path`` holds the ``$defs`` keys of the models between the top-level model and ``model``. When a
    nested object's key is already in ``path``, the function writes ``REPEATED_MODEL_LINE`` with the
    model's name and stops there.
    """
    required = set(model.get("required", ()))
    lines = []
    for name, field in model["properties"].items():
        status = "required" if name in required else "optional"
        text = field.get("description") or field.get("title")
        terms = ", ".join([_type_name(field, defs), *_constraints(field, defs), status])
        lines.append(f"{indent}{name} ({terms})" + (f": {text}" if text else ""))
        nested = _nested_model(field, defs)
        if nested is None:
            continue
        key, inner = nested
        inner_indent = indent + NESTED_FIELD_INDENT
        if key is not None and key in path:
            lines.append(inner_indent + REPEATED_MODEL_LINE.format(name=inner.get("title", key)))
        else:
            inner_path = path if key is None else path | {key}
            lines += _lead_lines(inner, inner_indent) + _field_lines(inner, defs, inner_indent, inner_path)
    return lines


def schema_instruction(output_type: type[BaseModel]) -> str:
    """Render ``output_type``'s fields as the plain-text instruction for the final user turn.

    The text opens with ``SCHEMA_INSTRUCTION_HEADER`` and the model's docstring. Each field then gets a
    line in the form ``<name> (<type>, <constraints>, required|optional): <description>``, for example
    ``notes (string, at most 1000 characters, required): Free-form notes.`` The constraints come from
    the field's JSON schema keywords, put into words:

    - ``minLength``/``maxLength`` and ``minItems``/``maxItems`` as counts of characters or items,
    - ``minimum``/``maximum`` and their exclusive forms as numeric bounds,
    - ``enum`` and ``const`` as the allowed JSON values,
    - ``pattern`` as the regex to match.

    For a list, the constraints on its items follow with an ``each`` prefix. A union whose members
    carry constraints of their own renders them as alternatives, each member's constraints joined by
    ``and``, as in ``exactly "ready" or exactly "pending"``. A field without constraints renders as
    ``<name> (<type>, required|optional)``. A field without a description falls back to its title.

    Below a field holding a model, whether directly, in a list or as an optional, the function writes
    the model's docstring and its fields one indent further in. It recurses the same way through models
    at any depth::

        files (list of object, required): The files to write.
          A file the draft writes.
          location (object, required)
            Where the file goes.
            path (string, required): An absolute output path.

    On the way down it tracks the models between the top level and the current field. When a model
    turns up again inside itself, the function writes a line such as ``Section (same fields as above)``
    in place of its fields, so a self-referential model renders in finite text.
    """
    schema = output_type.model_json_schema()
    defs = schema.get("$defs", {})
    root = _definition(schema, defs)
    root_name = _ref_name(schema)
    path = frozenset() if root_name is None else frozenset({root_name})
    return "\n".join([SCHEMA_INSTRUCTION_HEADER, *_lead_lines(root, ""), *_field_lines(root, defs, "", path)])


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
