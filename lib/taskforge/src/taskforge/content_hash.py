# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON encoding and sha256 content hashes over it.

``canonical_json`` is the encoding Taskforge hashes. It sorts keys, uses compact separators and
writes UTF-8 text, with bytes as base64 and dataclasses or pydantic models field by field.
``digest`` returns the sha256 of that encoding. ``pretty_json`` produces the indented form for
prompts and for files people read.
"""

import hashlib
import json

from pydantic_core import to_jsonable_python


def _jsonable(value: object) -> object:
    return to_jsonable_python(value, bytes_mode="base64")


def canonical_json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def pretty_json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def digest(value: object) -> str:
    """The sha256 hex of ``canonical_json(value)``."""
    return sha256_hex(canonical_json(value).encode())
