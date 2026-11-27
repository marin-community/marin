# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure, bounded remediation plans with stable evidence-based identities."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Protocol

from finestore.rl.curation import SourceUse
from finestore.rl.difficulty import PROTOCOL_ID, Solver
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import Backend

from experiments.post_training.task_curation.difficulty.ledger import digest, repair_verification_sources
from experiments.post_training.task_curation.difficulty.legacy import gap_priority
from experiments.post_training.task_curation.difficulty.state import (
    CatalogSource,
    CellState,
    CellStatus,
    Gap,
    GapReason,
    LedgerRow,
)
from experiments.post_training.task_curation.difficulty_config import DATA_ACCOUNT, SNOWBALL_VLLM_RELEASE_TAG
from experiments.post_training.task_curation.difficulty_output_budget import require_solver_protocol

MAX_TASKS_PER_JOB = 20
MAX_GPU_HOURS_PER_JOB = 40.0
DECLARATIVE_EVIDENCE = "daytona:data:declarative"
PANEL_INVENTORY_REVISION = "2026-11-20-census-artifact-record"


@dataclass(frozen=True)
class SourceAttempts:
    source_id: str
    registry_revision: str
    tasks: tuple[tuple[str, str, int], ...]
    queue_uri: str
    queue_sha256: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class AttemptBatch:
    solver: Solver
    sources: tuple[SourceAttempts, ...]
    n_attempts: int
    max_gpu_hours: float = MAX_GPU_HOURS_PER_JOB
    vllm_release_tag: str | None = None
    daytona_account: str | None = None
    solver_protocol_id: str = PROTOCOL_ID

    def __post_init__(self) -> None:
        require_solver_protocol(self.solver, self.solver_protocol_id)


@dataclass(frozen=True)
class RebuildEnvironmentArtifact:
    uri: str
    cells: tuple[tuple[str, Solver], ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class BuildSnapshot:
    source: SourceAttempts
    solver: Solver


@dataclass(frozen=True)
class BuildTaskSpecFactory:
    source: SourceAttempts
    solver: Solver


@dataclass(frozen=True)
class FreezeSamplePanel:
    source: SourceAttempts
    cells: tuple[tuple[str, Solver], ...]
    declaration_json: str
    inventory_revision: str | None = None


@dataclass(frozen=True)
class RecheckExecutabilityAdmission:
    source: SourceAttempts
    cells: tuple[tuple[str, Solver], ...]


@dataclass(frozen=True)
class QualifyJudgeRoute:
    model: str
    cells: tuple[tuple[str, Solver], ...]
    evidence_ids: tuple[str, ...]
    requested_temperature: float | None = None


@dataclass(frozen=True)
class RetryAfter:
    provider: str
    source: SourceAttempts
    solver: Solver


@dataclass(frozen=True)
class Escalate:
    reason: GapReason
    evidence: tuple[str, ...]
    source: SourceAttempts
    solver: Solver
    detail: str | None


Action = (
    AttemptBatch
    | RebuildEnvironmentArtifact
    | BuildSnapshot
    | BuildTaskSpecFactory
    | FreezeSamplePanel
    | RecheckExecutabilityAdmission
    | QualifyJudgeRoute
    | RetryAfter
    | Escalate
)


def action_payload(action: Action) -> dict:
    payload = {"kind": type(action).__name__, **asdict(action)}
    # Unchanged solver routes retain their existing in-flight action identities.
    if isinstance(action, AttemptBatch) and action.vllm_release_tag is None:
        del payload["vllm_release_tag"]
    if isinstance(action, AttemptBatch) and action.daytona_account is None:
        del payload["daytona_account"]
    if isinstance(action, AttemptBatch) and action.solver_protocol_id == PROTOCOL_ID:
        del payload["solver_protocol_id"]
    if isinstance(action, FreezeSamplePanel) and action.inventory_revision is None:
        del payload["inventory_revision"]
    return payload


def action_id(action: Action) -> str:
    return digest(action_payload(action))


def load_action(uri: str) -> Action:
    """Read a manifest and reject bytes that do not match its content identity."""
    path = StoragePath(uri)
    payload = json.loads(path.read_text())
    if path.name != digest(payload) + ".json":
        raise ValueError("Action manifest changed after planning")
    return action_from_payload(payload)


def action_gpu_hour_reservation(action: Action) -> float:
    if isinstance(action, AttemptBatch):
        return action.max_gpu_hours
    return MAX_GPU_HOURS_PER_JOB if isinstance(action, QualifyJudgeRoute) else 0.0


def action_cells(action: Action) -> tuple[tuple[str, Solver], ...]:
    if isinstance(action, AttemptBatch):
        return tuple((source.source_id, action.solver) for source in action.sources)
    if isinstance(action, (BuildSnapshot, BuildTaskSpecFactory, RetryAfter, Escalate)):
        return ((action.source.source_id, action.solver),)
    return action.cells


def selected_source(cell: CellState) -> SourceAttempts:
    return SourceAttempts(
        cell.source.source_id,
        cell.source.registry_revision,
        cell.missing,
        cell.source.queue_uri,
        cell.source.queue_sha256,
        cell.evidence_ids,
    )


class Remediation(Protocol):
    def plan(self, cells: Sequence[CellState]) -> list[Action]: ...


@dataclass(frozen=True)
class AttemptRemediation:
    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        actions: list[Action] = []
        for cell in cells:
            source = selected_source(cell)
            # A source can require several bounded jobs. Each owns disjoint tasks.
            for start in range(0, len(source.tasks), MAX_TASKS_PER_JOB):
                tasks = source.tasks[start : start + MAX_TASKS_PER_JOB]
                selected = SourceAttempts(
                    source.source_id,
                    source.registry_revision,
                    tasks,
                    source.queue_uri,
                    source.queue_sha256,
                    source.evidence_ids,
                )
                release = SNOWBALL_VLLM_RELEASE_TAG if cell.solver == Solver.SNOWBALL else None
                actions.append(
                    AttemptBatch(
                        cell.solver,
                        (selected,),
                        sum(n for _, _, n in tasks),
                        vllm_release_tag=release,
                        daytona_account=DATA_ACCOUNT if cell.source.backend == Backend.DAYTONA else None,
                        solver_protocol_id=cell.solver_protocol_id,
                    )
                )
        return actions


@dataclass(frozen=True)
class ProviderRemediation:
    available: frozenset[Solver] = frozenset()

    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        actions: list[Action] = []
        for cell in cells:
            if cell.solver in self.available:
                actions.extend(AttemptRemediation().plan([cell]))
            else:
                actions.append(RetryAfter(cell.solver.value, selected_source(cell), cell.solver))
        return actions


@dataclass(frozen=True)
class ArtifactRemediation:
    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        grouped: dict[str, list[CellState]] = {}
        actions: list[Action] = []
        for cell in cells:
            for gap in cell.gaps:
                if gap.reason != GapReason.ENVIRONMENT_ARTIFACT_MISSING:
                    continue
                if not gap.artifact_uri or not gap.artifact_uri.startswith("s3://"):
                    actions.append(
                        Escalate(
                            gap.reason,
                            cell.evidence_ids,
                            selected_source(cell),
                            cell.solver,
                            f"Relocate and bind original artifact to regional S3: {gap.artifact_uri}",
                        )
                    )
                    continue
                grouped.setdefault(gap.artifact_uri, []).append(cell)
        for uri, group in sorted(grouped.items()):
            actions.append(
                RebuildEnvironmentArtifact(
                    uri,
                    tuple((c.source.source_id, c.solver) for c in group),
                    tuple(sorted({e for c in group for e in c.evidence_ids})),
                )
            )
        return actions


@dataclass(frozen=True)
class SnapshotRemediation:
    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        return [BuildSnapshot(selected_source(cell), cell.solver) for cell in cells]


@dataclass(frozen=True)
class TaskSpecRemediation:
    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        actions: list[Action] = []
        missing: dict[CatalogSource, list[CellState]] = {}
        for cell in cells:
            if cell.source.queue_uri:
                actions.append(BuildTaskSpecFactory(selected_source(cell), cell.solver))
            else:
                missing.setdefault(cell.source, []).append(cell)
        for source, group in missing.items():
            if source.declaration_json is None:
                raise ValueError(f"Missing frozen declaration for panel repair: {source.source_id}")
            actions.append(
                FreezeSamplePanel(
                    selected_source(group[0]),
                    tuple(sorted((cell.source.source_id, cell.solver) for cell in group)),
                    source.declaration_json,
                    PANEL_INVENTORY_REVISION,
                )
            )
        return actions


@dataclass(frozen=True)
class AdmissionRemediation:
    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        grouped: dict[CatalogSource, list[CellState]] = {}
        actions: list[Action] = []
        for cell in cells:
            grouped.setdefault(cell.source, []).append(cell)
        for source, group in grouped.items():
            actions.append(
                RecheckExecutabilityAdmission(
                    SourceAttempts(
                        source.source_id,
                        source.registry_revision,
                        tuple((task, sha, 1) for task, sha in source.tasks),
                        source.queue_uri,
                        source.queue_sha256,
                        tuple(sorted({evidence for cell in group for evidence in cell.evidence_ids})),
                    ),
                    tuple(sorted({(cell.source.source_id, cell.solver) for cell in group})),
                )
            )
        return actions


@dataclass(frozen=True)
class JudgeRemediation:
    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        grouped: dict[tuple[str, float | None], list[CellState]] = {}
        actions: list[Action] = []
        for cell in cells:
            for gap in cell.gaps:
                if gap.reason != GapReason.JUDGE_ROUTE_UNQUALIFIED:
                    continue
                if gap.judge_model:
                    grouped.setdefault((gap.judge_model, gap.judge_temperature), []).append(cell)
                else:
                    actions.append(
                        Escalate(
                            gap.reason,
                            cell.evidence_ids,
                            selected_source(cell),
                            cell.solver,
                            "Judge identity is missing from legacy evidence",
                        )
                    )
        for (model, temperature), group in sorted(grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))):
            actions.append(
                QualifyJudgeRoute(
                    model,
                    tuple((c.source.source_id, c.solver) for c in group),
                    tuple(sorted({e for c in group for e in c.evidence_ids})),
                    temperature,
                )
            )
        return actions


@dataclass(frozen=True)
class EscalationRemediation:
    reason: GapReason

    def plan(self, cells: Sequence[CellState]) -> list[Action]:
        return [
            Escalate(
                self.reason,
                cell.evidence_ids,
                selected_source(cell),
                cell.solver,
                next((gap.capability for gap in cell.gaps if gap.reason == self.reason), None),
            )
            for cell in cells
        ]


REMEDIATIONS: dict[GapReason, Remediation] = {
    GapReason.COVERAGE_SHORTFALL: AttemptRemediation(),
    GapReason.NO_MEASUREMENT_ROW: AttemptRemediation(),
    GapReason.GRADER_INFRASTRUCTURE_FAILURE: AttemptRemediation(),
    GapReason.ENVIRONMENT_START_FAILURE: AttemptRemediation(),
    GapReason.PROVIDER_CAPACITY_WAIT: ProviderRemediation(),
    GapReason.ENVIRONMENT_ARTIFACT_MISSING: ArtifactRemediation(),
    GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING: SnapshotRemediation(),
    GapReason.TASK_SPEC_FACTORY_PENDING: TaskSpecRemediation(),
    GapReason.JUDGE_ROUTE_UNQUALIFIED: JudgeRemediation(),
    GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED: EscalationRemediation(GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED),
    GapReason.NETWORK_POLICY_VIOLATION: EscalationRemediation(GapReason.NETWORK_POLICY_VIOLATION),
    GapReason.UNSUPPORTED_COMPOSE_DOCUMENT: EscalationRemediation(GapReason.UNSUPPORTED_COMPOSE_DOCUMENT),
    GapReason.EXECUTABILITY_ADMISSION_PENDING: AdmissionRemediation(),
    GapReason.SERVING_RELEASE_MISMATCH: EscalationRemediation(GapReason.SERVING_RELEASE_MISMATCH),
    GapReason.SERVING_PROBE_MISMATCH: EscalationRemediation(GapReason.SERVING_PROBE_MISMATCH),
    GapReason.SOURCE_QUEUE_VALIDATION_FAILED: EscalationRemediation(GapReason.SOURCE_QUEUE_VALIDATION_FAILED),
    GapReason.CONTROLLER_ACTION_CANCELLED: EscalationRemediation(GapReason.CONTROLLER_ACTION_CANCELLED),
    GapReason.CONTROLLER_ACTION_FAILED: EscalationRemediation(GapReason.CONTROLLER_ACTION_FAILED),
}


def plan(
    cells: Sequence[CellState],
    ledger: Sequence[LedgerRow],
    available_providers: frozenset[Solver] = frozenset(),
    available_artifacts: frozenset[str] = frozenset(),
    ledger_root: str | None = None,
) -> list[Action]:
    """Return only actions whose exact evidence has not already been acted on."""
    recorded = {row.action_id for row in ledger if row.action_id}
    grouped: dict[GapReason, list[CellState]] = {}
    for cell in sorted(cells, key=lambda c: (c.solver, c.source.source_id)):
        if cell.state in {CellStatus.MEASURED, CellStatus.TERMINAL} or cell.job_ids:
            continue
        obsolete_snapshot = cell.source.backend == Backend.DAYTONA and any(
            gap.reason == GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING for gap in cell.gaps
        )
        remaining = tuple(
            gap
            for gap in cell.gaps
            if not (gap.reason == GapReason.ENVIRONMENT_ARTIFACT_MISSING and gap.artifact_uri in available_artifacts)
            and not (gap.reason == GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING and obsolete_snapshot)
        )
        if remaining != cell.gaps:
            cell = replace(
                cell,
                gaps=remaining,
                evidence_ids=cell.evidence_ids
                + ((DECLARATIVE_EVIDENCE,) if obsolete_snapshot else ())
                + tuple(
                    sorted(
                        "available:" + uri
                        for uri in available_artifacts
                        if any(gap.artifact_uri == uri for gap in cell.gaps)
                    )
                ),
            )
        hard = tuple(
            gap
            for gap in remaining
            if gap.reason
            not in {GapReason.COVERAGE_SHORTFALL, GapReason.PROVIDER_CAPACITY_WAIT, GapReason.NO_MEASUREMENT_ROW}
        )
        for gap in hard or remaining or (Gap(GapReason.COVERAGE_SHORTFALL),):
            grouped.setdefault(gap.reason, []).append(cell)
    actions: dict[str, tuple[GapReason, Action]] = {}
    for reason, selected in sorted(grouped.items()):
        remediation = REMEDIATIONS[reason]
        if isinstance(remediation, ProviderRemediation):
            remediation = replace(remediation, available=available_providers)
        for action in remediation.plan(selected):
            key = action_id(action)
            if key not in recorded:
                actions[key] = (reason, action)
    # Typed priority outranks digest order: a gap's repair verification plans before the
    # coverage-shortfall backlog regardless of where its action id sorts. A cell linked to a
    # recorded repair verification (its escalation landed and its repair is deployed) verifies
    # at the front of its priority band.
    verified = repair_verification_sources(ledger_root) if ledger_root else frozenset()
    train_sources = {cell.source.source_id for cell in cells if cell.source.intended_use == SourceUse.TRAIN}
    ordered = sorted(
        actions.items(),
        key=lambda item: (
            0 if any(source_id in train_sources for source_id, _ in action_cells(item[1][1])) else 1,
            gap_priority(item[1][0]),
            0 if repair_verifies(item[1], verified) else 1,
            item[1][0],
            item[0],
        ),
    )
    return [action for _, (_, action) in ordered]


def repair_verifies(linked: tuple[GapReason, Action], verified: frozenset[str]) -> bool:
    """Whether this action verifies a recorded repair: its cell is named by the repair's linkage."""
    _, action = linked
    return isinstance(action, AttemptBatch) and any(source.source_id in verified for source in action.sources)


def source_attempts_from_payload(value: dict) -> SourceAttempts:
    return SourceAttempts(
        value["source_id"],
        value["registry_revision"],
        tuple(tuple(task) for task in value["tasks"]),
        value["queue_uri"],
        value["queue_sha256"],
        tuple(value["evidence_ids"]),
    )


def action_from_payload(payload: dict) -> Action:
    """Decode a manifest into its typed action, retaining frozen source identities."""
    kind = payload["kind"]
    if kind == "AttemptBatch":
        return AttemptBatch(
            Solver(payload["solver"]),
            tuple(source_attempts_from_payload(s) for s in payload["sources"]),
            payload["n_attempts"],
            payload["max_gpu_hours"],
            payload.get("vllm_release_tag"),
            payload.get("daytona_account"),
            payload.get("solver_protocol_id", PROTOCOL_ID),
        )
    if kind == "RebuildEnvironmentArtifact":
        return RebuildEnvironmentArtifact(
            payload["uri"], tuple((s, Solver(m)) for s, m in payload["cells"]), tuple(payload["evidence_ids"])
        )
    if kind == "QualifyJudgeRoute":
        return QualifyJudgeRoute(
            payload["model"],
            tuple((s, Solver(m)) for s, m in payload["cells"]),
            tuple(payload["evidence_ids"]),
            payload.get("requested_temperature"),
        )
    if kind == "BuildSnapshot":
        return BuildSnapshot(source_attempts_from_payload(payload["source"]), Solver(payload["solver"]))
    if kind == "BuildTaskSpecFactory":
        return BuildTaskSpecFactory(source_attempts_from_payload(payload["source"]), Solver(payload["solver"]))
    if kind == "FreezeSamplePanel":
        return FreezeSamplePanel(
            source_attempts_from_payload(payload["source"]),
            tuple((source, Solver(solver)) for source, solver in payload["cells"]),
            payload["declaration_json"],
            payload.get("inventory_revision"),
        )
    if kind == "RecheckExecutabilityAdmission":
        return RecheckExecutabilityAdmission(
            source_attempts_from_payload(payload["source"]),
            tuple((source, Solver(solver)) for source, solver in payload["cells"]),
        )
    if kind == "RetryAfter":
        return RetryAfter(
            payload["provider"], source_attempts_from_payload(payload["source"]), Solver(payload["solver"])
        )
    if kind == "Escalate":
        return Escalate(
            GapReason(payload["reason"]),
            tuple(payload["evidence"]),
            source_attempts_from_payload(payload["source"]),
            Solver(payload["solver"]),
            payload["detail"],
        )
    raise ValueError(f"Unknown action kind: {kind}")
