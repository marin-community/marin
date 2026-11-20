# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist a campaign budget across controller restarts."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.ledger import LEDGER_VERSION
from experiments.post_training.task_curation.difficulty.planner import action_from_payload, action_gpu_hours
from experiments.post_training.task_curation.difficulty.state import LedgerRow, Outcome


@dataclass(frozen=True)
class GpuBudget:
    limit_gpu_hours: float
    started_at: str

    def recorded_gpu_hours(self, rows: Sequence[LedgerRow]) -> float:
        return sum(row.gpu_hours for row in rows if row.recorded_at >= self.started_at)

    def reserved_gpu_hours(self, rows: Sequence[LedgerRow], ledger: str) -> float:
        ended = {row.action_id for row in rows if row.outcome == Outcome.FINISHED}
        active = {row.action_id for row in rows if row.outcome == Outcome.SUBMITTED and row.action_id not in ended}
        total = 0.0
        for key in active:
            manifest = StoragePath(ledger) / LEDGER_VERSION / "actions" / f"{key}.json"
            total += action_gpu_hours(action_from_payload(json.loads(manifest.read_text())))
        return total


def campaign_budget(ledger: str, limit_gpu_hours: float) -> GpuBudget:
    """Bind the budget once under the reconciliation lease; restarts retain spend."""
    if limit_gpu_hours <= 0:
        raise ValueError("GPU-hour budget must be positive")
    target = StoragePath(ledger) / LEDGER_VERSION / "budget.json"
    if target.exists():
        budget = GpuBudget(**json.loads(target.read_text()))
        if budget.limit_gpu_hours != limit_gpu_hours:
            raise ValueError("Existing campaign budget differs; obtain a new budget ruling")
        return budget
    budget = GpuBudget(limit_gpu_hours, datetime.now(UTC).isoformat())
    target.parent.mkdirs()
    target.write_text(json.dumps(asdict(budget), sort_keys=True) + "\n")
    return budget
