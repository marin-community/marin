# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode searchable metadata without losing the original trace bytes."""

import json
import zlib


def searchable_metadata(value: object) -> object:
    """Escape NUL characters for PostgreSQL JSONB's searchable copy."""
    if isinstance(value, str):
        return value.replace("\x00", "\\u0000")
    if isinstance(value, dict):
        return {searchable_metadata(key): searchable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [searchable_metadata(item) for item in value]
    return value


def case_record(row: dict) -> dict:
    payload = row.pop("metadata_payload")
    if payload is not None:
        row.update(json.loads(zlib.decompress(bytes(payload))))
    return row
