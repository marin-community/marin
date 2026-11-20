# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute bounded plans under a regional lease and retain notification evidence."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from finestore.rl.difficulty import Solver
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.ledger import append_rows, digest, read_rows
from experiments.post_training.task_curation.difficulty.planner import (
    Action,
    AttemptBatch,
    BuildSnapshot,
    BuildTaskSpecFactory,
    Escalate,
    QualifyJudgeRoute,
    RebuildEnvironmentArtifact,
    RetryAfter,
    action_from_payload,
    action_id,
    action_payload,
    plan,
)
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog
from experiments.post_training.task_curation.difficulty.state import (
    CatalogSource,
    CoveragePolicy,
    Gap,
    GapReason,
    LedgerRow,
    Outcome,
)

CONCURRENT_JOB_CAP = 22
ATTEMPT_JOB_SLOTS = 8
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    job_id: str
    state: str
    solver: Solver | None
    slots: int = 1

    @property
    def active(self) -> bool:
        return self.state not in {"succeeded", "failed", "cancelled", "killed", "worker_failed", "unschedulable"}


class JobBackend(Protocol):
    def jobs(self) -> list[Job]: ...
    def submit(self, action: Action, manifest_uri: str) -> str: ...


class NotificationSink(Protocol):
    def write(self, key: str, message: str) -> None: ...


@dataclass(frozen=True)
class EscalationLog:
    root: str

    def write(self, key: str, message: str) -> None:
        """Publish a deduplicated escalation and its watchdog log event."""
        target = StoragePath(self.root) / "v1" / "escalations" / f"{key}.json"
        payload = {"action_id": key, "message": message}
        target.parent.mkdirs()
        if not target.exists():
            target.write_text(json.dumps(payload, sort_keys=True) + "\n")
        document = StoragePath(self.root) / "ESCALATIONS.md"
        text = document.read_text() if document.exists() else "# Difficulty escalations\n\n"
        if key not in text:
            document.write_text(text + f"- {key}: {message}\n")
        LOGGER.warning("DIFFICULTY_ESCALATION %s", json.dumps(payload, sort_keys=True))


def action_job_slots(action: Action) -> int:
    return ATTEMPT_JOB_SLOTS if isinstance(action, (AttemptBatch, QualifyJudgeRoute)) else 1


def action_cells(action: Action) -> tuple[tuple[str, Solver], ...]:
    payload = action_payload(action)
    if isinstance(action, AttemptBatch):
        return tuple((source.source_id, action.solver) for source in action.sources)
    if "source" in payload:
        return ((payload["source"]["source_id"], Solver(payload["solver"])),)
    return tuple((source_id, Solver(solver)) for source_id, solver in payload["cells"])


def action_rows(
    action: Action, catalog: Sequence[CatalogSource], outcome: Outcome, job_id: str | None, timestamp: str
) -> list[LedgerRow]:
    sources = {source.source_id: source for source in catalog}
    key = action_id(action)
    return [
        LedgerRow(
            sources[source_id].registry_revision,
            source_id,
            None,
            solver,
            digest([key, source_id, solver, outcome]),
            outcome,
            action.reason if isinstance(action, Escalate) else None,
            key,
            0.0,
            timestamp,
            action_id=key,
            job_id=job_id,
        )
        for source_id, solver in action_cells(action)
    ]


def execute_plan(
    actions: Sequence[Action],
    rows: Sequence[LedgerRow],
    catalog: Sequence[CatalogSource],
    ledger: str,
    backend: JobBackend,
    notifications: NotificationSink,
) -> dict:
    """Persist submissions after successful submit; a lost reply is recovered by job name."""
    jobs = backend.jobs()
    slots = sum(job.slots for job in jobs if job.active)
    recorded = {row.action_id for row in rows if row.action_id}
    submitted, escalated, deferred = [], [], []
    for action in actions:
        key = action_id(action)
        if key in recorded:
            continue
        if isinstance(action, RetryAfter):
            deferred.append(key)
            continue
        if isinstance(action, Escalate):
            notifications.write(
                key,
                f"Difficulty {action.reason}: {action.source.source_id} / {action.solver}; "
                f"{action.detail or 'protocol or source ruling required'}; evidence {','.join(action.evidence)}",
            )
            append_rows(ledger, action_rows(action, catalog, Outcome.ESCALATED, None, datetime.now(UTC).isoformat()))
            escalated.append(key)
            continue
        required_slots = action_job_slots(action)
        # Do not overlap a legacy campaign's solver while its receipts are live.
        if isinstance(action, AttemptBatch) and any(
            job.active and job.solver == action.solver and "-reconcile-" not in job.job_id for job in jobs
        ):
            deferred.append(key)
            continue
        if slots + required_slots > CONCURRENT_JOB_CAP:
            deferred.append(key)
            continue
        manifest = StoragePath(ledger) / "v1" / "actions" / f"{key}.json"
        payload = json.dumps(action_payload(action), sort_keys=True) + "\n"
        manifest.parent.mkdirs()
        if manifest.exists() and manifest.read_text() != payload:
            raise ValueError(f"Action identity collision: {manifest}")
        if not manifest.exists():
            manifest.write_text(payload)
        job_id = backend.submit(action, str(manifest))
        append_rows(ledger, action_rows(action, catalog, Outcome.SUBMITTED, job_id, datetime.now(UTC).isoformat()))
        LOGGER.info("DIFFICULTY_SUBMITTED %s", json.dumps({"action_id": key, "job_id": job_id}))
        submitted.append(job_id)
        slots += required_slots
    return {
        "submitted": submitted,
        "escalated": escalated,
        "deferred": deferred,
        "active_job_slots": slots,
        "concurrent_job_cap": CONCURRENT_JOB_CAP,
    }


def reconcile_once(
    ledger: str,
    catalog: Sequence[CatalogSource],
    policy: CoveragePolicy,
    backend: JobBackend,
    notifications: NotificationSink,
    available_providers: frozenset[Solver],
    available_artifacts: frozenset[str],
) -> dict:
    """Take one consistent evidence snapshot, execute its plan, then release the lease."""
    if not (StoragePath(ledger) / "v1" / "imports" / "*.json").glob():
        raise ValueError("Initial backfill is unsealed; finish import before reconcile")
    lease = create_lock(str(StoragePath(ledger) / "v1" / "reconcile.lock"))
    if not lease.try_acquire():
        return {"state": "another_reconciler_holds_lease"}
    try:
        with lease_refresh(lease):
            rows = read_rows(ledger)
            rows += collect_finished(rows, backend.jobs(), ledger, notifications)
            cells = cells_for_catalog(rows, catalog, policy)
            actions = plan(cells, rows, available_providers, available_artifacts)
            return execute_plan(actions, rows, catalog, ledger, backend, notifications)
    finally:
        lease.release()


def collect_finished(
    rows: Sequence[LedgerRow], jobs: Sequence[Job], ledger: str, notifications: NotificationSink
) -> list[LedgerRow]:
    """Apply completed repair evidence to the original submitted population only."""
    by_job = {job.job_id: job for job in jobs}
    ended = {row.action_id for row in rows if row.outcome == Outcome.FINISHED}
    additions = []
    for row in rows:
        if row.outcome != Outcome.SUBMITTED or row.action_id in ended:
            continue
        job = by_job.get(row.job_id)
        if job is None or job.active:
            continue
        timestamp = datetime.now(UTC).isoformat()
        additions.append(
            LedgerRow(
                row.registry_revision,
                row.source_id,
                None,
                row.solver,
                digest([row.attempt_id, "finished"]),
                Outcome.FINISHED,
                None,
                row.run_id,
                0.0,
                timestamp,
                action_id=row.action_id,
                job_id=row.job_id,
            )
        )
        target = StoragePath(ledger) / "v1" / "results" / f"{row.action_id}.json"
        if job.state == "succeeded":
            result = json.loads(target.read_text())
            if result["status"] == "completed":
                continue
            if result["status"] == "ready":
                manifest = StoragePath(ledger) / "v1" / "actions" / f"{row.action_id}.json"
                action = action_from_payload(json.loads(manifest.read_text()))
                repaired = repaired_gap(action)
                additions.append(
                    LedgerRow(
                        row.registry_revision,
                        row.source_id,
                        None,
                        row.solver,
                        digest([row.attempt_id, "repair-ready"]),
                        Outcome.REPAIRED,
                        repaired.reason,
                        row.run_id,
                        0.0,
                        timestamp,
                        artifact_uri=repaired.artifact_uri,
                        judge_model=repaired.judge_model,
                        judge_temperature=repaired.judge_temperature,
                        evidence_uri=str(target),
                    )
                )
                continue
            reason = GapReason(result["reason"])
            detail = result["detail"]
        else:
            reason, detail = GapReason.EXECUTABILITY_ADMISSION_PENDING, f"Iris child {row.job_id}: {job.state}"
        key = digest([row.action_id, row.source_id, row.solver, "repair-unavailable"])
        notifications.write(key, f"Difficulty {reason}: {row.source_id} / {row.solver}; {detail}; evidence {target}")
        additions.append(
            LedgerRow(
                row.registry_revision,
                row.source_id,
                None,
                row.solver,
                digest([row.attempt_id, "escalated"]),
                Outcome.ESCALATED,
                reason,
                row.run_id,
                0.0,
                timestamp,
                action_id=row.action_id,
                job_id=row.job_id,
                capability=detail,
                evidence_uri=str(target),
            )
        )
    append_rows(ledger, additions)
    return additions


def repaired_gap(action: Action) -> Gap:
    """Identify the one cause a successful repair can clear."""
    if isinstance(action, RebuildEnvironmentArtifact):
        return Gap(GapReason.ENVIRONMENT_ARTIFACT_MISSING, artifact_uri=action.uri)
    if isinstance(action, QualifyJudgeRoute):
        return Gap(
            GapReason.JUDGE_ROUTE_UNQUALIFIED,
            judge_model=action.model,
            judge_temperature=action.requested_temperature,
        )
    if isinstance(action, BuildSnapshot):
        return Gap(GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING)
    if isinstance(action, BuildTaskSpecFactory):
        return Gap(GapReason.TASK_SPEC_FACTORY_PENDING)
    raise ValueError(f"Action cannot publish repair evidence: {type(action).__name__}")
