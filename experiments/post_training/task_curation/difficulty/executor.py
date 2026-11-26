# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute bounded plans under a regional lease and retain notification evidence."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol

from finestore.rl.difficulty import Solver
from iris.resources.state import JobState, is_job_finished
from marin.external_dependencies import VLLM_GPU_RELEASE
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.budget import GpuBudget, campaign_budget, record_pool_cost
from experiments.post_training.task_curation.difficulty.campaign import TPU_SERVING_ID, campaign_record
from experiments.post_training.task_curation.difficulty.ledger import (
    ACTIONS_DIRECTORY,
    ESCALATIONS_DIRECTORY,
    IMPORTS_DIRECTORY,
    LEDGER_VERSION,
    RESULTS_DIRECTORY,
    append_rows,
    digest,
    read_rows,
)
from experiments.post_training.task_curation.difficulty.planner import (
    Action,
    AttemptBatch,
    BuildSnapshot,
    BuildTaskSpecFactory,
    Escalate,
    FreezeSamplePanel,
    QualifyJudgeRoute,
    RebuildEnvironmentArtifact,
    RecheckExecutabilityAdmission,
    RetryAfter,
    action_cells,
    action_from_payload,
    action_gpu_hour_reservation,
    action_id,
    action_payload,
    plan,
)
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog
from experiments.post_training.task_curation.difficulty.source_repairs import record_source_repair
from experiments.post_training.task_curation.difficulty.state import (
    CatalogSource,
    CoveragePolicy,
    Gap,
    GapReason,
    LedgerRow,
    Outcome,
)
from experiments.post_training.task_curation.difficulty_config import SELF_HOSTED_SOLVERS

CONCURRENT_JOB_CAP = 64
REPAIR_JOB_SLOT_CAP = 8
ATTEMPT_JOB_SLOTS = 4
ATTEMPT_JOBS_PER_SOLVER = MappingProxyType(
    {
        Solver.SMALL: 4,
        Solver.LARGE: 4,
        Solver.SNOWBALL: 4,
        Solver.HOSTED: 2,
    }
)
# Reserve 16 provider workers; split the remaining 112 in the 128/64/64-chip ratio.
TPU_ATTEMPT_JOBS_PER_SOLVER = MappingProxyType({Solver.SMALL: 7, Solver.LARGE: 14, Solver.SNOWBALL: 7, Solver.HOSTED: 4})
TPU_CONCURRENT_JOB_CAP = sum(TPU_ATTEMPT_JOBS_PER_SOLVER.values()) * ATTEMPT_JOB_SLOTS + REPAIR_JOB_SLOT_CAP
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    job_id: str
    state: JobState
    solver: Solver | None
    slots: int = 1

    @property
    def active(self) -> bool:
        return not is_job_finished(self.state)


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
        target = StoragePath(self.root) / LEDGER_VERSION / ESCALATIONS_DIRECTORY / f"{key}.json"
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
    budget: GpuBudget | None = None,
) -> dict:
    """Persist submissions after successful submit; a lost reply is recovered by job name."""
    jobs = list(backend.jobs())
    slots = sum(job.slots for job in jobs if job.active)
    repair_slots = sum(job.slots for job in jobs if job.active and job.solver is None)
    recorded = {row.action_id for row in rows if row.action_id}
    submitted, escalated, deferred = [], [], []
    ended = {row.action_id for row in rows if row.outcome == Outcome.FINISHED}
    claimed: set[tuple[str, str]] = {
        (row.source_id, row.solver.value)
        for row in rows
        if row.outcome == Outcome.SUBMITTED and row.job_id and row.action_id not in ended
    }
    spent = budget.recorded_gpu_hours(rows) if budget else 0.0
    reserved = budget.reserved_gpu_hours(rows, ledger) if budget else 0.0
    tpu_campaign = campaign_record(ledger) is not None
    concurrent_job_cap = TPU_CONCURRENT_JOB_CAP if tpu_campaign else CONCURRENT_JOB_CAP
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
        if isinstance(action, AttemptBatch):
            solver_cap = (TPU_ATTEMPT_JOBS_PER_SOLVER if tpu_campaign else ATTEMPT_JOBS_PER_SOLVER)[action.solver]
            if sum(job.active and job.solver == action.solver for job in jobs) >= solver_cap:
                deferred.append(key)
                continue
            cells = {(source.source_id, action.solver.value) for source in action.sources}
            if cells & claimed:
                # One in-flight AttemptBatch per cell: concurrent batches would double the
                # solver reservation without adding scored receipts.
                deferred.append(key)
                continue
        if slots + required_slots > concurrent_job_cap:
            deferred.append(key)
            continue
        if not isinstance(action, AttemptBatch) and repair_slots + required_slots > REPAIR_JOB_SLOT_CAP:
            deferred.append(key)
            continue
        reservation = action_gpu_hour_reservation(action)
        if (
            budget
            and budget.limit_gpu_hours is not None
            and reservation
            and spent + reserved + reservation > budget.limit_gpu_hours
        ):
            budget_key = digest([budget.started_at, budget.limit_gpu_hours, "budget-exhausted"])
            if not (StoragePath(ledger) / LEDGER_VERSION / ESCALATIONS_DIRECTORY / f"{budget_key}.json").exists():
                notifications.write(
                    budget_key,
                    f"Difficulty GPU budget exhausted: recorded {spent:.6f} + in-flight {reserved:.6f} "
                    f"+ next reservation {reservation:.6f} exceeds {budget.limit_gpu_hours:.6f} GPU-h; "
                    "new GPU jobs are deferred; an operator budget ruling is required",
                )
            deferred.append(key)
            continue
        manifest = StoragePath(ledger) / LEDGER_VERSION / ACTIONS_DIRECTORY / f"{key}.json"
        payload = json.dumps(action_payload(action), sort_keys=True) + "\n"
        manifest.parent.mkdirs()
        if manifest.exists() and manifest.read_text() != payload:
            raise ValueError(f"Action identity collision: {manifest}")
        if not manifest.exists():
            manifest.write_text(payload)
        job_id = backend.submit(action, str(manifest))
        append_rows(ledger, action_rows(action, catalog, Outcome.SUBMITTED, job_id, datetime.now(UTC).isoformat()))
        if isinstance(action, AttemptBatch):
            claimed |= {(source.source_id, action.solver.value) for source in action.sources}
        LOGGER.info("DIFFICULTY_SUBMITTED %s", json.dumps({"action_id": key, "job_id": job_id}))
        submitted.append(job_id)
        if isinstance(action, AttemptBatch):
            jobs.append(Job(job_id, JobState.PENDING, action.solver, required_slots))
        else:
            repair_slots += required_slots
        slots += required_slots
        reserved += reservation
    return {
        "submitted": submitted,
        "escalated": escalated,
        "deferred": deferred,
        "active_job_slots": slots,
        "concurrent_job_cap": concurrent_job_cap,
        "recorded_gpu_hours": spent,
        "reserved_gpu_hours": reserved,
        "budget_gpu_hours": budget.limit_gpu_hours if budget else None,
    }


def reconcile_once(
    ledger: str,
    catalog: Sequence[CatalogSource],
    policy: CoveragePolicy,
    backend: JobBackend,
    notifications: NotificationSink,
    available_providers: frozenset[Solver],
    available_artifacts: frozenset[str],
    budget_limit_gpu_hours: float | None,
    serving_pool: str | None = None,
) -> dict:
    """Take one consistent evidence snapshot, execute its plan, then release the lease."""
    if not (StoragePath(ledger) / LEDGER_VERSION / IMPORTS_DIRECTORY / "*.json").glob():
        raise ValueError("Initial backfill is unsealed; finish import before reconcile")
    lease = create_lock(str(StoragePath(ledger) / LEDGER_VERSION / "reconcile.lock"))
    if not lease.try_acquire():
        return {"state": "another_reconciler_holds_lease"}
    try:
        with lease_refresh(lease):
            rows = read_rows(ledger)
            rows += collect_finished(rows, backend.jobs(), ledger, notifications)
            tpu_campaign = campaign_record(ledger) is not None
            if serving_pool is not None:
                pools = available_providers if tpu_campaign else frozenset(SELF_HOSTED_SOLVERS)
                rows += record_pool_cost(ledger, serving_pool, rows, solvers=pools)
            budget = campaign_budget(ledger, budget_limit_gpu_hours)
            cells = cells_for_catalog(rows, catalog, policy)
            actions = plan(cells, rows, available_providers, available_artifacts, ledger_root=ledger)
            if serving_pool is not None:
                if tpu_campaign:
                    actions = [
                        action
                        for action in actions
                        if not isinstance(action, AttemptBatch) or action.solver in available_providers
                    ]
                for index, action in enumerate(actions):
                    if not isinstance(action, AttemptBatch):
                        continue
                    release = (
                        VLLM_GPU_RELEASE.release_tag if action.solver == Solver.SNOWBALL else action.vllm_release_tag
                    )
                    if tpu_campaign and action.solver != Solver.HOSTED:
                        release = TPU_SERVING_ID
                    actions[index] = replace(action, max_gpu_hours=0.0, vllm_release_tag=release)
            return execute_plan(actions, rows, catalog, ledger, backend, notifications, budget)
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
        target = StoragePath(ledger) / LEDGER_VERSION / RESULTS_DIRECTORY / f"{row.action_id}.json"
        error_identity = None
        if job.state == JobState.SUCCEEDED and target.exists():
            result = json.loads(target.read_text())
            if result["status"] == "completed":
                continue
            if result["status"] == "ready":
                manifest = StoragePath(ledger) / LEDGER_VERSION / ACTIONS_DIRECTORY / f"{row.action_id}.json"
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
            handoff_uri = record_source_repair(ledger, row.action_id, result, str(target))
            if handoff_uri is not None:
                LOGGER.info("DIFFICULTY_SOURCE_REPAIR %s", handoff_uri)
                continue
        else:
            reason = (
                GapReason.CONTROLLER_ACTION_CANCELLED
                if job.state == JobState.KILLED
                else GapReason.CONTROLLER_ACTION_FAILED
            )
            detail = f"Iris child {row.job_id}: {job.state}"
            error_identity = f"iris_child_{job.state}"
            if job.state == JobState.SUCCEEDED:
                detail = f"Iris child {row.job_id} succeeded without a result in this campaign ledger"
                error_identity = "completed_job_missing_campaign_result"
            if target.exists():
                failure = json.loads(target.read_text())
                if failure["status"] == "failed":
                    reason, detail = GapReason(failure["reason"]), failure["detail"]
                    error_identity = failure["error_identity"]
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
                error_identity=error_identity,
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
    if isinstance(action, (BuildTaskSpecFactory, FreezeSamplePanel)):
        return Gap(GapReason.TASK_SPEC_FACTORY_PENDING)
    if isinstance(action, RecheckExecutabilityAdmission):
        return Gap(GapReason.EXECUTABILITY_ADMISSION_PENDING)
    raise ValueError(f"Action cannot publish repair evidence: {type(action).__name__}")
