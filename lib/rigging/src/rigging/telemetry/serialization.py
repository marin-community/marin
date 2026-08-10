# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded serialization for direct telemetry records."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

MAX_ATTRIBUTES = 64
MAX_STRING_LENGTH = 4_096

type EventValue = str | int | float | bool


@dataclass(frozen=True)
class EventBody:
    """A flat, typed event payload."""

    fields: Mapping[str, EventValue]


def json_bytes(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def json_bytes_bounded(value: object, limit: int) -> bytes:
    encoded = bytearray()
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"), sort_keys=True)
    for chunk in encoder.iterencode(value):
        chunk_bytes = chunk.encode()
        if len(encoded) + len(chunk_bytes) > limit:
            raise ValueError("encoded telemetry record exceeds the batch limit")
        encoded.extend(chunk_bytes)
    return bytes(encoded)


def event_fields(body: EventBody, budget: int) -> dict[str, EventValue]:
    fields = dict(body.fields)
    if len(fields) > MAX_ATTRIBUTES:
        raise ValueError(f"event bodies may contain at most {MAX_ATTRIBUTES} fields")
    for key, value in fields.items():
        validate_string(key, "event field")
        if isinstance(value, str):
            validate_string(value, "event value")
        elif isinstance(value, bool):
            continue
        elif isinstance(value, int):
            if not -(1 << 63) <= value <= (1 << 64) - 1:
                raise ValueError("event integer is outside serde_json's exact range")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("event numbers must be finite")
        else:
            raise ValueError("event values must be strings, integers, floats, or booleans")
    json_bytes_bounded(fields, budget)
    return fields


def validate_string(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_STRING_LENGTH
        or len(value.encode()) > MAX_STRING_LENGTH
    ):
        raise ValueError(f"{field} must be a nonempty string of at most {MAX_STRING_LENGTH} bytes")


def validate_attributes(attributes: Mapping[str, str]) -> None:
    if len(attributes) > MAX_ATTRIBUTES:
        raise ValueError(f"attributes may contain at most {MAX_ATTRIBUTES} entries")
    for key, value in attributes.items():
        validate_string(key, "attribute key")
        validate_string(value, "attribute value")
