# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure coverage state and the closed ledger vocabulary."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from finestore.rl.curation import SourceUse
from finestore.rl.difficulty import PROTOCOL_ID, Solver

from experiments.post_training.task_curation.difficulty.resource_deviations import ResourceDeviation
from experiments.post_training.task_curation.difficulty_output_budget import require_solver_protocol

REPAIR_BLOCKED = "blocked"


class GapReason(StrEnum):
    COVERAGE_SHORTFALL = "COVERAGE_SHORTFALL"
    PROVIDER_CAPACITY_WAIT = "PROVIDER_CAPACITY_WAIT"
    PROTOCOL_INPUT_BUDGET_EXCEEDED = "PROTOCOL_INPUT_BUDGET_EXCEEDED"
    NETWORK_POLICY_VIOLATION = "NETWORK_POLICY_VIOLATION"
    DAYTONA_SNAPSHOT_PREFLIGHT_PENDING = "DAYTONA_SNAPSHOT_PREFLIGHT_PENDING"
    TASK_SPEC_FACTORY_PENDING = "TASK_SPEC_FACTORY_PENDING"
    ENVIRONMENT_ARTIFACT_MISSING = "ENVIRONMENT_ARTIFACT_MISSING"
    ENVIRONMENT_START_FAILURE = "ENVIRONMENT_START_FAILURE"
    JUDGE_ROUTE_UNQUALIFIED = "JUDGE_ROUTE_UNQUALIFIED"
    GRADER_INFRASTRUCTURE_FAILURE = "GRADER_INFRASTRUCTURE_FAILURE"
    UNSUPPORTED_COMPOSE_DOCUMENT = "UNSUPPORTED_COMPOSE_DOCUMENT"
    EXECUTABILITY_ADMISSION_PENDING = "EXECUTABILITY_ADMISSION_PENDING"
    SERVING_RELEASE_MISMATCH = "SERVING_RELEASE_MISMATCH"
    SERVING_PROBE_MISMATCH = "SERVING_PROBE_MISMATCH"
    SOURCE_QUEUE_VALIDATION_FAILED = "SOURCE_QUEUE_VALIDATION_FAILED"
    CONTROLLER_ACTION_CANCELLED = "CONTROLLER_ACTION_CANCELLED"
    CONTROLLER_ACTION_FAILED = "CONTROLLER_ACTION_FAILED"
    NO_MEASUREMENT_ROW = "NO_MEASUREMENT_ROW"


class ActionFailure(ValueError):
    """A child failure with a specific ledger cause."""

    def __init__(self, reason: GapReason, detail: str):
        super().__init__(detail)
        self.reason = reason


class Outcome(StrEnum):
    SCORED = "scored"
    UNSCORED = "unscored"
    OBSERVATION = "observation"
    COST = "cost"
    SUBMITTED = "submitted"
    FINISHED = "finished"
    ESCALATED = "escalated"
    REPAIRED = "repaired"
    PROTOCOL_SUPERSEDED = "protocol_superseded"


class ArtifactKind(StrEnum):
    REQUIREMENTS_LOCK = "requirements_lock"
    PREPARED_IMAGE = "prepared_image"
    DOCKER_BUILD = "docker_build"


@dataclass(frozen=True)
class Gap:
    reason: GapReason
    artifact_uri: str | None = None
    judge_model: str | None = None
    capability: str | None = None
    artifact_kind: ArtifactKind | None = None
    judge_temperature: float | None = None
    # The typed identity of the infrastructure failure, for example http_401
    # from the judge transport evidence. Never free text.
    error_identity: str | None = None


@dataclass(frozen=True)
class AdmissionBlocker:
    task_id: str
    task_sha256: str
    reason: str


@dataclass(frozen=True)
class LedgerRow:
    registry_revision: str
    source_id: str
    task_sha256: str | None
    solver: Solver
    attempt_id: str
    outcome: Outcome
    reason: GapReason | None
    run_id: str
    gpu_hours: float
    recorded_at: str
    task_id: str | None = None
    reward: float | None = None
    artifact_uri: str | None = None
    judge_model: str | None = None
    capability: str | None = None
    evidence_uri: str | None = None
    action_id: str | None = None
    job_id: str | None = None
    # Legacy observations are audit evidence, never scored attempts.
    legacy_status: str | None = None
    artifact_kind: ArtifactKind | None = None
    judge_temperature: float | None = None
    error_identity: str | None = None
    secondary_gaps: tuple[Gap, ...] = ()
    queue_sha256: str | None = None
    resource_deviation: ResourceDeviation | None = None
    solver_protocol_id: str = PROTOCOL_ID
    supersedes_attempt_id: str | None = None

    def __post_init__(self) -> None:
        require_solver_protocol(self.solver, self.solver_protocol_id)
        deviation = self.resource_deviation
        if deviation is not None and (self.source_id, self.task_id, self.task_sha256) != (
            deviation.source_id,
            deviation.task_id,
            deviation.task_sha256,
        ):
            raise ValueError("Ledger resource deviation belongs to another frozen task")

    @property
    def gap(self) -> Gap | None:
        if self.reason is None:
            return None
        return Gap(
            self.reason,
            self.artifact_uri,
            self.judge_model,
            self.capability,
            self.artifact_kind,
            self.judge_temperature,
            self.error_identity,
        )


@dataclass(frozen=True)
class CatalogSource:
    source_id: str
    registry_revision: str
    tasks: tuple[tuple[str, str], ...]
    queue_uri: str
    queue_sha256: str
    declaration_json: str | None = None
    backend: str | None = None
    intended_use: SourceUse | None = None


@dataclass(frozen=True)
class PanelReference:
    source_id: str
    registry_revision: str
    queue_uri: str
    queue_sha256: str


@dataclass(frozen=True)
class TerminalRuling:
    source_id: str
    solver: Solver
    reason: GapReason
    registry_revision: str
    evidence_uri: str


@dataclass(frozen=True)
class CoveragePolicy:
    attempts_per_task: int = 1
    terminal_rulings: tuple[TerminalRuling, ...] = ()
    large_protocol_id: str = PROTOCOL_ID

    def __post_init__(self) -> None:
        require_solver_protocol(Solver.LARGE, self.large_protocol_id)

    def solver_protocol_id(self, solver: Solver) -> str:
        return self.large_protocol_id if solver == Solver.LARGE else PROTOCOL_ID


class CellStatus(StrEnum):
    MEASURED = "MEASURED"
    NEEDS_ATTEMPTS = "NEEDS_ATTEMPTS"
    BLOCKED = "BLOCKED"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class CellState:
    source: CatalogSource
    solver: Solver
    state: CellStatus
    missing: tuple[tuple[str, str, int], ...]
    gaps: tuple[Gap, ...]
    evidence_ids: tuple[str, ...]
    job_ids: tuple[str, ...] = ()
    resource_deviations: tuple[ResourceDeviation, ...] = ()
    solver_protocol_id: str = PROTOCOL_ID

    @property
    def n_missing(self) -> int:
        return sum(count for _, _, count in self.missing)


PROTOCOL_DEPENDENT_GAPS = frozenset(
    {
        GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED,
        GapReason.PROVIDER_CAPACITY_WAIT,
        GapReason.SERVING_RELEASE_MISMATCH,
        GapReason.SERVING_PROBE_MISMATCH,
        GapReason.CONTROLLER_ACTION_CANCELLED,
        GapReason.CONTROLLER_ACTION_FAILED,
    }
)


def coverage_evidence(row: LedgerRow, protocol_id: str) -> LedgerRow | None:
    """Carry source conditions and in-flight holds across protocols, never old scores."""
    if row.solver_protocol_id == protocol_id:
        return row
    original = ((row.gap,) if row.gap else ()) + row.secondary_gaps
    shared = tuple(gap for gap in original if gap.reason not in PROTOCOL_DEPENDENT_GAPS)
    if row.outcome == Outcome.SCORED:
        if not shared:
            return None
        row = replace(row, outcome=Outcome.OBSERVATION, reward=None)
    if not original or shared == original:
        return row
    if not shared:
        return None
    first = shared[0]
    return replace(
        row,
        reason=first.reason,
        artifact_uri=first.artifact_uri,
        judge_model=first.judge_model,
        capability=first.capability,
        artifact_kind=first.artifact_kind,
        judge_temperature=first.judge_temperature,
        error_identity=first.error_identity,
        secondary_gaps=shared[1:],
    )


def cell_state(ledger: Sequence[LedgerRow], source: CatalogSource, solver: Solver, policy: CoveragePolicy) -> CellState:
    """Count distinct scored receipts for the current task bytes and population.

    Old population evidence stays in the ledger. A changed release needs fresh
    measurement unless its identity has an explicit carry-forward upstream.
    Provider waits never contribute to the scored count.
    """
    protocol_id = policy.solver_protocol_id(solver)
    rows = {
        row.attempt_id: row
        for original in ledger
        if (row := coverage_evidence(original, protocol_id)) is not None
        if row.source_id == source.source_id
        and row.solver == solver
        and row.registry_revision == source.registry_revision
    }
    conditions = {
        row.resource_deviation.version if row.resource_deviation is not None else None
        for row in rows.values()
        if row.outcome == Outcome.SCORED and (row.task_id, row.task_sha256) in source.tasks
    }
    if len(conditions) > 1:
        raise ValueError("A difficulty cell cannot mix scored execution-resource conditions")
    missing = []
    for task_id, sha in source.tasks:
        n = sum(
            row.outcome == Outcome.SCORED and row.task_sha256 == sha and row.task_id == task_id for row in rows.values()
        )
        if n < policy.attempts_per_task:
            missing.append((task_id, sha, policy.attempts_per_task - n))
    jobs = tuple(
        sorted(
            {
                row.job_id
                for row in rows.values()
                if row.outcome == Outcome.SUBMITTED
                and row.job_id
                and not any(end.action_id == row.action_id and end.outcome == Outcome.FINISHED for end in rows.values())
            }
        )
    )
    source_observations = [
        row for row in rows.values() if row.outcome == Outcome.OBSERVATION and row.task_sha256 is None
    ]
    latest = max((row.recorded_at for row in source_observations), default=None)
    relevant = [row for row in source_observations if row.recorded_at == latest]
    # An unrelated task that finishes later must not hide a blocking cause.
    for task_id, sha in source.tasks:
        attempts = [
            row
            for row in rows.values()
            if row.task_sha256 == sha
            and row.task_id == task_id
            and row.outcome in {Outcome.SCORED, Outcome.UNSCORED, Outcome.OBSERVATION}
        ]
        if attempts:
            last = max(attempts, key=lambda row: (row.recorded_at, row.attempt_id))
            if last.outcome != Outcome.SCORED:
                relevant.append(last)
    gaps = tuple(
        sorted(
            {
                gap
                for row in relevant
                for gap in ((row.gap,) if row.gap else ()) + row.secondary_gaps
                if not any(
                    repair.outcome == Outcome.REPAIRED
                    and repair.recorded_at > row.recorded_at
                    and repair.reason == gap.reason
                    and repair.artifact_uri == gap.artifact_uri
                    and repair.judge_model == gap.judge_model
                    and repair.judge_temperature == gap.judge_temperature
                    for repair in rows.values()
                )
            },
            key=lambda gap: (gap.reason, gap.artifact_uri or "", gap.judge_model or "", gap.capability or ""),
        )
    )
    state = CellStatus.NEEDS_ATTEMPTS
    policy_violation = any(gap.reason == GapReason.NETWORK_POLICY_VIOLATION for gap in gaps)
    if not missing and source.tasks and not policy_violation:
        state, gaps = CellStatus.MEASURED, ()
    elif not relevant:
        gaps = (
            Gap(
                GapReason.COVERAGE_SHORTFALL
                if any(row.outcome == Outcome.SCORED for row in rows.values())
                else GapReason.NO_MEASUREMENT_ROW
            ),
        )
    elif any(
        gap.reason
        not in {
            GapReason.COVERAGE_SHORTFALL,
            GapReason.PROVIDER_CAPACITY_WAIT,
            GapReason.NO_MEASUREMENT_ROW,
            GapReason.GRADER_INFRASTRUCTURE_FAILURE,
            # Infrastructure retry: the wave re-runs the attempt once the machine pulls.
            GapReason.ENVIRONMENT_START_FAILURE,
        }
        for gap in gaps
    ):
        state = CellStatus.BLOCKED
    if not source.tasks:
        state, gaps = CellStatus.BLOCKED, (Gap(GapReason.TASK_SPEC_FACTORY_PENDING),)
    if gaps and all(
        any(
            ruling.source_id == source.source_id
            and ruling.solver == solver
            and ruling.registry_revision == source.registry_revision
            and ruling.reason == gap.reason
            and ruling.evidence_uri
            for ruling in policy.terminal_rulings
        )
        for gap in gaps
    ):
        state = CellStatus.TERMINAL
    return CellState(
        source,
        solver,
        state,
        tuple(missing),
        gaps,
        tuple(
            sorted(
                row.attempt_id
                for row in rows.values()
                if row.outcome not in {Outcome.SUBMITTED, Outcome.FINISHED, Outcome.ESCALATED}
            )
        ),
        jobs,
        tuple(
            sorted(
                {
                    row.resource_deviation
                    for row in rows.values()
                    if row.resource_deviation is not None and (row.task_id, row.task_sha256) in source.tasks
                },
                key=lambda deviation: (deviation.version, deviation.task_id),
            )
        ),
        protocol_id,
    )
