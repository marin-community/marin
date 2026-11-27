# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable identities for JSON source records and pipeline inputs."""

import hashlib
import json
from typing import Any

from taskcompendium.models import TaskSpec


def canonical_sha256(row: dict[str, Any]) -> str:
    """Hash UTF-8 JSON with sorted keys and compact separators."""
    document = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(document.encode()).hexdigest()


def task_key(source_id: str | None, task: TaskSpec) -> tuple[str | None, str, str, str]:
    """Identify an upstream row within its pinned revision and curated source."""
    return source_id, task.source.dataset, task.source.revision, task.source.row
