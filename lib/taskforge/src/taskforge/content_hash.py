# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON encoding and sha256 content hashes over it.

``canonical_json`` is the encoding Taskforge hashes. It sorts keys, uses compact separators and
writes UTF-8 text, with bytes as base64 and dataclass or pydantic values field by field. A set is
written as a list sorted by each element's canonical JSON, since its iteration order follows the
process's hash seed; this holds for sets in dicts, lists, tuples and dataclasses, not for sets held by
a pydantic model, which pydantic serializes itself.
``digest`` returns the sha256 of that encoding. ``pretty_json`` produces the indented form for
prompts and for files people read.
"""

import dataclasses
import hashlib
import json

from pydantic_core import to_jsonable_python


def _jsonable(value: object) -> object:
    return to_jsonable_python(_ordered(value), bytes_mode="base64")


def _ordered(value: object) -> object:
    """``value`` with every set it holds in plain containers or dataclasses replaced by a sorted list."""
    if isinstance(value, set | frozenset):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, dict):
        return {key: _ordered(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_ordered(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and not hasattr(value, "__pydantic_fields__"):
        return {field.name: _ordered(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return value


def canonical_json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def pretty_json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def digest(value: object) -> str:
    """The sha256 hex of ``canonical_json(value)``."""
    return sha256_hex(canonical_json(value).encode())
