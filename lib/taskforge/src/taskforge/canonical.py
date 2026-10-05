# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON, content digests, and atomic file replacement.

``canonical_json`` is the one encoding Taskforge hashes: sorted keys, compact separators, UTF-8
text, bytes as base64, and dataclasses and pydantic models by field. ``digest`` is the sha256 of
it, so a cache key, a catalog hash and a step memo key agree on the same value. ``pretty_json`` is
the indented form shown to models and written for people to read.
"""

import hashlib
import json
import os
from pathlib import Path

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


def write_atomic(path: Path, content: bytes) -> None:
    """Replace ``path`` with ``content`` through a per-process temporary file beside it."""
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(content)
    temp.replace(path)
