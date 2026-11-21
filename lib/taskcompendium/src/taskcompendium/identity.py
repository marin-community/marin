# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable identities for JSON source records and pipeline inputs."""

import hashlib
import json
from typing import Any


def canonical_sha256(row: dict[str, Any]) -> str:
    """Hash UTF-8 JSON with sorted keys and compact separators."""
    document = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(document.encode()).hexdigest()
