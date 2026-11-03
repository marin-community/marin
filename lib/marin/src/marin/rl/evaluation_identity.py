# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check supplied evaluation membership and protocol identities."""

import hashlib
import json
from typing import Any


def validate_evaluation_identity(
    *,
    task_ids: list[str] | None,
    task_count: int | None,
    membership_sha256: str | None,
    protocol: dict[str, Any] | None,
    protocol_sha256: str | None,
) -> None:
    """Reject supplied identity mismatches while preserving unknown historical fields."""
    if task_ids is not None:
        membership = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
        if len(set(task_ids)) != len(task_ids) or membership != membership_sha256:
            raise ValueError("Evaluation membership does not match its recorded identity")
        if task_count is not None and task_count != len(task_ids):
            raise ValueError("Evaluation task count disagrees with membership")
    if protocol is not None:
        encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(encoded.encode()).hexdigest() != protocol_sha256:
            raise ValueError("Evaluation protocol does not match its recorded identity")
