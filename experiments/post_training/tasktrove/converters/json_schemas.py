# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repairs and checks for the LLM-generated JSON Schemas the Nemotron structured-output templates ship.

``required`` shows up as ``{"group": [...]}`` or a bare ``true``, ``additionalProperties`` as the
string ``"false"`` or a stray list, and numeric keywords as ``null``. ``normalize_schema`` repairs
the shapes common enough to fix without guessing at intent; ``schema_error`` reports what is still
broken after that, so those tasks are rejected rather than shipped with a grader that always raises.
"""

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

# Numeric/length keywords the dataset sometimes sets to JSON null instead of omitting; Draft
# 2020-12 requires a number here, so null always fails the metaschema check.
_NULLABLE_NUMERIC_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "minimum",
        "maximum",
        "multipleOf",
    }
)


def _flatten_required(value: object) -> list[str] | None:
    """``required`` as a proper string list, or ``None`` when the shape can't be trusted.

    Handles the dataset's non-standard ``{"group": ["a", "b"]}`` form (flattened to ``["a",
    "b"]``) alongside the standard list. Any other shape (seen in practice: a bare ``true``) is
    dropped rather than guessed at.
    """
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, dict):
        flat: list[str] = []
        for group in value.values():
            if isinstance(group, list):
                flat.extend(item for item in group if isinstance(item, str))
            elif isinstance(group, str):
                flat.append(group)
        return flat
    return None


def _normalize_additional_properties(value: object) -> object | None:
    """``additionalProperties`` as a bool or nested schema, or ``None`` to drop it.

    ``"false"``/``"true"`` strings are the dataset's most common miswrite and are coerced; a list
    or free-text description (also seen) can't be turned into a schema or boolean without
    guessing, so the keyword is dropped and additional properties are simply left unrestricted.
    """
    if isinstance(value, bool | dict):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def normalize_schema(node: object) -> object:
    if isinstance(node, dict):
        out: dict = {}
        for key, value in node.items():
            if key == "required":
                flattened = _flatten_required(value)
                if flattened is not None:
                    out[key] = flattened
            elif key == "additionalProperties":
                normalized = _normalize_additional_properties(value)
                if normalized is not None:
                    out[key] = normalize_schema(normalized) if isinstance(normalized, dict) else normalized
            elif key in _NULLABLE_NUMERIC_KEYWORDS and value is None:
                continue
            else:
                out[key] = normalize_schema(value)
        return out
    if isinstance(node, list):
        return [normalize_schema(item) for item in node]
    return node


def is_trivial(schema: dict) -> bool:
    """True when the schema imposes no content constraint a candidate could fail meaningfully.

    Covers the object schemas the dataset ships with empty ``properties`` and ``required`` (the
    only passing document is the fixed ``{}``, regardless of the source text) and the equivalent
    empty-``items`` array shape.
    """
    schema_type = schema.get("type")
    if schema_type in (None, "object"):
        return not schema.get("properties") and not schema.get("required")
    if schema_type == "array":
        items = schema.get("items")
        return not items
    return False


def schema_error(schema: dict) -> str | None:
    """The metaschema violation in ``schema``, or ``None`` when it is a valid JSON Schema."""
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as error:
        return error.message
    return None
