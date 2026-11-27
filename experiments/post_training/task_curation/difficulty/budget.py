# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist a campaign budget across controller restarts."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.ledger import (
    ACTIONS_DIRECTORY,
    LEDGER_VERSION,
    append_rows,
    digest,
)
from experiments.post_training.task_curation.difficulty.planner import action_from_payload, action_gpu_hour_reservation
from experiments.post_training.task_curation.difficulty.state import LedgerRow, Outcome
from experiments.post_training.task_curation.difficulty_config import SELF_HOSTED_SOLVERS


@dataclass(frozen=True)
class GpuBudget:
    limit_gpu_hours: float | None
    started_at: str

    def recorded_gpu_hours(self, rows: Sequence[LedgerRow]) -> float:
        return sum(row.gpu_hours for row in rows if row.recorded_at >= self.started_at)

    def reserved_gpu_hours(self, rows: Sequence[LedgerRow], ledger: str) -> float:
        ended = {row.action_id for row in rows if row.outcome == Outcome.FINISHED}
        active = {row.action_id for row in rows if row.outcome == Outcome.SUBMITTED and row.action_id not in ended}
        total = 0.0
        for key in active:
            manifest = StoragePath(ledger) / LEDGER_VERSION / ACTIONS_DIRECTORY / f"{key}.json"
            total += action_gpu_hour_reservation(action_from_payload(json.loads(manifest.read_text())))
        return total


def campaign_budget(ledger: str, limit_gpu_hours: float | None) -> GpuBudget:
    """Bind the budget once under the reconciliation lease; restarts retain spend."""
    if limit_gpu_hours is not None and limit_gpu_hours <= 0:
        raise ValueError("GPU-hour budget must be positive")
    target = StoragePath(ledger) / LEDGER_VERSION / "budget.json"
    if target.exists():
        budget = GpuBudget(**json.loads(target.read_text()))
        if limit_gpu_hours is None and budget.limit_gpu_hours is not None:
            budget = GpuBudget(None, budget.started_at)
            target.write_text(json.dumps(asdict(budget), sort_keys=True) + "\n")
        if budget.limit_gpu_hours != limit_gpu_hours:
            raise ValueError("Existing campaign budget differs; obtain a new budget ruling")
        return budget
    budget = GpuBudget(limit_gpu_hours, datetime.now(UTC).isoformat())
    target.parent.mkdirs()
    target.write_text(json.dumps(asdict(budget), sort_keys=True) + "\n")
    return budget


def record_pool_cost(
    ledger: str,
    registry: str,
    rows: Sequence[LedgerRow],
    *,
    solvers: frozenset[Solver] = frozenset(SELF_HOSTED_SOLVERS),
) -> list[LedgerRow]:
    """Account persistent allocation once, from warm readiness, across overlapping batches and restarts."""
    now = datetime.now(UTC)
    additions = []
    for solver in sorted(solvers & frozenset(SELF_HOSTED_SOLVERS)):
        receipt_uri = str(StoragePath(registry) / f"{solver.value}.json")
        receipt = json.loads(StoragePath(receipt_uri).read_text())
        source = f"engine-pool:{solver.value}"
        ready_at = datetime.fromisoformat(receipt["ready_at"])
        accounted_until = max(
            (
                datetime.fromisoformat(row.recorded_at)
                for row in rows
                if row.run_id == registry and row.source_id == source and row.outcome == Outcome.COST
            ),
            default=ready_at,
        )
        # A replacement must not inherit downtime before its own readiness.
        started_at = max(ready_at, accounted_until)
        start = started_at.isoformat()
        elapsed = (now - started_at).total_seconds()
        if elapsed <= 0:
            continue
        evidence_uri = receipt_uri
        if receipt.get("platform") == "tpu":
            accounting = {
                "solver": solver.value,
                "start": start,
                "end": now.isoformat(),
                "chip_count": receipt["chip_count"],
                "chip_hours": receipt["chip_count"] * elapsed / 3600,
                "receipt_uri": receipt_uri,
                "budget_enforced": False,
                "basis": "pool allocation wall time; includes recovery intervals",
            }
            path = StoragePath(ledger) / LEDGER_VERSION / "tpu-accounting" / (digest(accounting) + ".json")
            path.parent.mkdirs()
            path.write_text(json.dumps(accounting, sort_keys=True) + "\n")
            evidence_uri = str(path)
        additions.append(
            LedgerRow(
                receipt["revision"],
                source,
                None,
                solver,
                digest([registry, solver, start, now.isoformat()]),
                Outcome.COST,
                None,
                registry,
                receipt["gpu_count"] * elapsed / 3600,
                now.isoformat(),
                evidence_uri=evidence_uri,
            )
        )
    append_rows(ledger, additions)
    return additions
