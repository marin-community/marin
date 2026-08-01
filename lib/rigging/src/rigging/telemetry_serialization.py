# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded JSON validation for direct telemetry records."""

import json
import math
from collections.abc import Mapping
from typing import Any

MAX_ATTRIBUTES = 64
MAX_STRING_LENGTH = 4_096
MAX_EVENT_DEPTH = 32


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def json_bytes_bounded(value: Any, limit: int) -> bytes:
    encoded = bytearray()
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"), sort_keys=True)
    for chunk in encoder.iterencode(value):
        chunk_bytes = chunk.encode()
        if len(encoded) + len(chunk_bytes) > limit:
            raise ValueError("encoded telemetry record exceeds the batch limit")
        encoded.extend(chunk_bytes)
    return bytes(encoded)


def validate_event_body(value: Any, budget: int) -> None:
    if value is None:
        raise ValueError("event body must not be None")

    def consume(remaining: int, amount: int) -> int:
        if amount > remaining:
            raise ValueError("event body exceeds the record byte budget")
        return remaining - amount

    def string_size(item: Any, field: str, remaining: int, punctuation: int) -> int:
        if not isinstance(item, str) or (field == "event body key" and not item):
            raise ValueError(f"{field} must be a nonempty string")
        if len(item) > MAX_STRING_LENGTH:
            raise ValueError(f"{field} exceeds {MAX_STRING_LENGTH} bytes")
        if len(item) + punctuation > remaining:
            raise ValueError("event body exceeds the record byte budget")
        encoded_size = len(item.encode())
        if encoded_size > MAX_STRING_LENGTH:
            raise ValueError(f"{field} exceeds {MAX_STRING_LENGTH} bytes")
        return encoded_size + punctuation

    def visit(item: Any, depth: int, remaining_nodes: int, remaining_bytes: int) -> tuple[int, int]:
        if remaining_nodes <= 0:
            raise ValueError("event body exceeds the record node budget")
        if depth > MAX_EVENT_DEPTH:
            raise ValueError(f"event body exceeds JSON depth {MAX_EVENT_DEPTH}")
        remaining_nodes -= 1
        if isinstance(item, str):
            remaining_bytes = consume(remaining_bytes, string_size(item, "event body string", remaining_bytes, 2))
        elif item is None:
            remaining_bytes = consume(remaining_bytes, 4)
        elif isinstance(item, bool):
            remaining_bytes = consume(remaining_bytes, 5)
        elif isinstance(item, int):
            if not -(1 << 63) <= item <= (1 << 64) - 1:
                raise ValueError("event body integer is outside serde_json's exact range")
            remaining_bytes = consume(remaining_bytes, 20)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("event body numbers must be finite")
            remaining_bytes = consume(remaining_bytes, 32)
        elif isinstance(item, dict):
            remaining_bytes = consume(remaining_bytes, 2)
            for index, (key, child) in enumerate(item.items()):
                if remaining_nodes <= 0:
                    raise ValueError("event body exceeds the record node budget")
                remaining_nodes -= 1
                key_size = string_size(key, "event body key", remaining_bytes, 3 + bool(index))
                remaining_bytes = consume(remaining_bytes, key_size)
                remaining_nodes, remaining_bytes = visit(child, depth + 1, remaining_nodes, remaining_bytes)
        elif isinstance(item, list | tuple):
            remaining_bytes = consume(remaining_bytes, 2)
            for index, child in enumerate(item):
                if index:
                    remaining_bytes = consume(remaining_bytes, 1)
                remaining_nodes, remaining_bytes = visit(child, depth + 1, remaining_nodes, remaining_bytes)
        else:
            raise ValueError("event body must contain only JSON values")
        return remaining_nodes, remaining_bytes

    visit(value, 0, budget, budget)


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
