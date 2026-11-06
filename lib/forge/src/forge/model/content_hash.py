# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical tagged content hashes and ordinary JSON for human-readable files."""

import base64
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import PurePath

from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from taskcompendium.identity import canonical_sha256

BYTES_TAG = "__forge_bytes__"
MAPPING_TAG = "__forge_mapping__"


def json_value(value: object) -> object:
    """JSON boundary representation; binary payloads use a reversible tagged value."""
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, bytes):
        return {BYTES_TAG: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, BaseModel):
        return json_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {MAPPING_TAG: {key: json_value(item) for key, item in value.items()}}
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__qualname__}")


def digest(value: object) -> str:
    """Adapt typed event values to the shared TaskCompendium canonical hash."""
    return canonical_sha256({"value": json_value(value)})


def pretty_json(value: object) -> str:
    return json.dumps(
        to_jsonable_python(value, bytes_mode="base64"), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
    )
