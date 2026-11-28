# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable identities for JSON source records and pipeline inputs."""

import hashlib
import json
from typing import Any, Protocol


class TaskDocument(Protocol):
    def source_document(self, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]: ...


def task_spec_sha256(task: TaskDocument) -> str:
    """Hash the stored canonical source document, preserving its schema vintage."""
    return canonical_sha256(task.source_document())


def task_identity(task: TaskDocument) -> str:
    """Identify the complete public and private task contract, excluding provenance."""
    return canonical_sha256(task.source_document(exclude=TASK_IDENTITY_EXCLUDED_FIELDS))


TASK_IDENTITY_EXCLUDED_FIELDS = frozenset({"id", "source", "tags", "labels"})
"""The provenance fields task_identity excludes when it hashes a task contract."""


def canonical_sha256(row: dict[str, Any]) -> str:
    """Hash UTF-8 JSON with sorted keys and compact separators."""
    document = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(document.encode()).hexdigest()
