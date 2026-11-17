# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep immutable review and rollout content compact in the SQL gateway."""

import base64
import gzip
import hashlib
import io
import json
from typing import Any

MAX_CONTENT_BYTES = 32 * 1024 * 1024


def pack_content(content: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > MAX_CONTENT_BYTES:
        raise ValueError("One evidence record exceeds the bounded content viewer")
    return {
        "encoding": "gzip+base64",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "payload": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
    }


def unpack_content(record: dict[str, Any]) -> dict[str, Any]:
    if record["encoding"] != "gzip+base64":
        raise ValueError("Unknown staged evidence encoding")
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(record["payload"], validate=True))) as stream:
        raw = stream.read(MAX_CONTENT_BYTES + 1)
    if (
        len(raw) != record["bytes"]
        or len(raw) > MAX_CONTENT_BYTES
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("Staged evidence content failed its size or SHA-256 check")
    return json.loads(raw)
