# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe shared provider admission outside the task-attempt ledger."""

import json
from datetime import UTC, datetime

import httpx
from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

from experiments.post_training.task_curation.difficulty.ledger import LEDGER_VERSION, digest
from experiments.post_training.task_curation.difficulty.runtime import (
    TOGETHER_API_KEY_ENV,
    TOGETHER_BASE_URL,
    TOGETHER_MODEL,
    hosted_liveness_response,
)

CAPACITY_STATUSES = frozenset({408, 429, 503, 504})


def available_provider_solvers(ledger: str) -> frozenset[Solver]:
    """Admit retries only after the shared hosted route serves a liveness request."""
    try:
        response = hosted_liveness_response(TOGETHER_BASE_URL, require_secret(TOGETHER_API_KEY_ENV), TOGETHER_MODEL)
        status_code = response.status_code
        if status_code not in CAPACITY_STATUSES:
            response.raise_for_status()
    except httpx.TimeoutException:
        status_code = 408
    evidence = {
        "provider": "together",
        "status_code": status_code,
        "recorded_at": datetime.now(UTC).isoformat(),
        "task_attempt": False,
    }
    target = StoragePath(ledger) / LEDGER_VERSION / "providers" / (digest(evidence) + ".json")
    target.parent.mkdirs()
    target.write_text(json.dumps(evidence, sort_keys=True) + "\n")
    return frozenset(Solver) if status_code == 200 else frozenset()
