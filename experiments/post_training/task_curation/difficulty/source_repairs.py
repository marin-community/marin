# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain source-owned repair handoffs without requesting operator escalation."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from finestore.rl.difficulty import DifficultyGap
from pydantic import TypeAdapter
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.ledger import ACTIONS_DIRECTORY, LEDGER_VERSION, digest
from experiments.post_training.task_curation.difficulty.legacy import admission_cause
from experiments.post_training.task_curation.difficulty.planner import RecheckExecutabilityAdmission, load_action
from experiments.post_training.task_curation.difficulty.state import REPAIR_BLOCKED, AdmissionBlocker, LedgerRow, Outcome
from experiments.post_training.task_curation.difficulty_evidence import write_same_json_evidence

SOURCE_REPAIRS_DIRECTORY = "source-repairs"
TOOL_PROVIDER_SOURCES = frozenset(
    {f"Nemotron:nemotron_ultra_{family}/ultra_sft_step3200_math_tir" for family in ("mopd", "rlvr1", "rlvr2")}
)
SHARED_EXECUTOR_SOURCE = "TaskTrove:laion__nemotron-gym-arc-agi-python-inductive-v2"
SHARED_GRADER_SOURCES = frozenset(
    {"RST:laion/rts-1k-rootless-apptainer", "PrimeIntellect:PrimeIntellect/Multi-SWE-RL-Verified"}
)


class SourceRepairRoute(StrEnum):
    SCOPED_CONSUMER_GATE = "scoped_consumer_gate"
    SHARED_GRADER_OPTION_2 = "shared_grader_option_2"


@dataclass(frozen=True)
class SourceRepairHandoff:
    source_id: str
    registry_revision: str
    route: SourceRepairRoute
    cells: tuple[tuple[str, str], ...]
    evidence_uri: str
    blockers: tuple[AdmissionBlocker, ...]


def source_repair_route(source_id: str, blockers: tuple[AdmissionBlocker, ...]) -> SourceRepairRoute | None:
    """Route only verified blocked rechecks in the coordinator's source scope."""
    if not blockers:
        return None
    causes = {admission_cause(blocker.reason)[0] for blocker in blockers}
    if source_id in TOOL_PROVIDER_SOURCES and causes == {DifficultyGap.TOOL_PROVIDER_REQUIRED}:
        return SourceRepairRoute.SCOPED_CONSUMER_GATE
    if source_id == SHARED_EXECUTOR_SOURCE and causes == {DifficultyGap.SHARED_EXECUTOR_REQUIRED}:
        return SourceRepairRoute.SCOPED_CONSUMER_GATE
    if source_id in SHARED_GRADER_SOURCES and causes == {DifficultyGap.UNRESOLVED_DOCKER_BUILD}:
        return SourceRepairRoute.SHARED_GRADER_OPTION_2
    return None


@dataclass(frozen=True)
class AdmissionRepairProof:
    source_id: str
    registry_revision: str
    checked_tasks: int
    population_changed: bool
    blockers: tuple[AdmissionBlocker, ...]


def source_repair_handoff(
    action: RecheckExecutabilityAdmission, proof: AdmissionRepairProof, evidence_uri: str
) -> SourceRepairHandoff | None:
    """Validate the frozen proof and select its handoff, or None outside the repair scope."""
    route = source_repair_route(action.source.source_id, proof.blockers)
    if route is None:
        return None
    if (proof.source_id, proof.registry_revision) != (action.source.source_id, action.source.registry_revision):
        raise ValueError("Source repair result differs from its frozen admission population")
    if proof.checked_tasks != len(action.source.tasks) or proof.population_changed:
        raise ValueError("Source repair result does not cover its unchanged frozen sample")
    hashes = {task_id: sha for task_id, sha, _ in action.source.tasks}
    if any(hashes[blocker.task_id] != blocker.task_sha256 for blocker in proof.blockers):
        raise ValueError("Source repair blocker differs from its planned task bytes")
    return SourceRepairHandoff(
        action.source.source_id, action.source.registry_revision, route, action.cells, evidence_uri, proof.blockers
    )


def record_source_repair(ledger: str, action_id: str, result: dict, evidence_uri: str) -> str | None:
    """Return the verified handoff URI, or None when the result is outside its scope."""
    if result.get("status") != REPAIR_BLOCKED or not result.get("blockers"):
        return None
    manifest = StoragePath(ledger) / LEDGER_VERSION / ACTIONS_DIRECTORY / (action_id + ".json")
    action = load_action(str(manifest))
    if not isinstance(action, RecheckExecutabilityAdmission):
        return None
    proof = TypeAdapter(AdmissionRepairProof).validate_python(result)
    handoff = source_repair_handoff(action, proof, evidence_uri)
    if handoff is None:
        return None
    target = StoragePath(ledger) / LEDGER_VERSION / SOURCE_REPAIRS_DIRECTORY / (digest(asdict(handoff)) + ".json")
    target.parent.mkdirs(exist_ok=True)
    write_same_json_evidence(target, asdict(handoff))
    if TypeAdapter(SourceRepairHandoff).validate_json(target.read_text()) != handoff:
        raise ValueError(f"Source repair handoff readback differs: {target}")
    return str(target)


def record_completed_source_repairs(ledger: str, rows: list[LedgerRow]) -> list[str]:
    """Recover handoffs from rechecks completed by an older controller bundle."""
    candidates = {
        (row.action_id, row.evidence_uri)
        for row in rows
        if row.outcome == Outcome.ESCALATED
        and row.action_id
        and row.evidence_uri
        and row.source_id in TOOL_PROVIDER_SOURCES | SHARED_GRADER_SOURCES | {SHARED_EXECUTOR_SOURCE}
    }
    recorded = []
    for action_id, uri in sorted(candidates):
        path = StoragePath(uri)
        if not path.exists():
            continue  # Failed children have no completed gate result to route.
        target = record_source_repair(ledger, action_id, json.loads(path.read_text()), uri)
        if target is not None:
            recorded.append(target)
    return recorded


def source_repair_handoffs(ledger: str) -> list[SourceRepairHandoff]:
    return [
        TypeAdapter(SourceRepairHandoff).validate_json(path.read_text())
        for path in sorted((StoragePath(ledger) / LEDGER_VERSION / SOURCE_REPAIRS_DIRECTORY / "*.json").glob(), key=str)
    ]
