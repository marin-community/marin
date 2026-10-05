# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select qualified task rounds, track checkpoint progress, and seal resumable evidence."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256, pinned_bytes
from experiments.post_training.russell_rsi.sources import compact_json_sha256

ATTEMPTS_PER_TASK = 8
CALIBRATION_TEMPERATURE = 1.0
MINIMUM_TASKS = 16
MAX_PILOTS = 3
MAX_GLM_RESPONSES = 24
PILOT_UPDATES = 4


class Difficulty(StrEnum):
    EASY = "easy"
    INTERMEDIATE = "intermediate"
    HARD = "hard"


class StopReason(StrEnum):
    PILOT_LIMIT = "pilot_limit"
    NO_IMPROVEMENT = "no_champion_improvement"
    QUALIFICATION = "qualification_failure"
    REWARD_VARIATION = "reward_variation_failure"
    TASK_SUPPLY = "task_supply_exhausted"
    TRAINING_SIGNAL = "insufficient_training_signal"
    CALIBRATION_FAILURE = "calibration_correctness_failure"


class IncompleteCalibrationError(ValueError):
    """Calibration has valid evidence, but one or more grades are absent."""

    def __init__(self, missing_task_ids: tuple[str, ...]):
        self.missing_task_ids = missing_task_ids
        super().__init__("Calibration is missing one or more task grades")


@dataclass(frozen=True)
class QualifiedTask:
    task_id: str
    task_sha256: str
    admission_sha256: str
    source_id: str
    capability: str
    contract_id: str = ""
    relation: str = "independent"


@dataclass(frozen=True)
class Measurement:
    checkpoint_identity: str
    task_sha256: str
    rewards: tuple[float, ...]

    @property
    def difficulty(self) -> Difficulty:
        if len(self.rewards) != ATTEMPTS_PER_TASK:
            raise ValueError("Difficulty requires eight graded attempts")
        successes = sum(reward > 0 for reward in self.rewards)
        if successes >= 6:
            return Difficulty.EASY
        if successes >= 2:
            return Difficulty.INTERMEDIATE
        return Difficulty.HARD


def calibration_measurements(
    summary: dict, bank: tuple[QualifiedTask, ...], model_identity: str, bank_identity: str
) -> tuple[Measurement, ...]:
    """Require complete calibration evidence from the selected checkpoint and task bank."""
    if summary["model_identity"] != model_identity or summary["tasks_identity"] != bank_identity:
        raise ValueError("Calibration checkpoint or task-bank identity changed")
    rewards = summary["task_rewards"]
    if (
        summary["count"] != len(bank)
        or summary["samples_per_task"] != ATTEMPTS_PER_TASK
        or set(rewards) != {task.task_id for task in bank}
    ):
        raise ValueError("Calibration requires eight finite grades per qualified task")
    for group in rewards.values():
        if (
            not isinstance(group, list)
            or len(group) > ATTEMPTS_PER_TASK
            or any(
                type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1 for reward in group
            )
        ):
            raise ValueError("Calibration requires eight finite grades per qualified task")
    missing = tuple(task.task_id for task in bank if len(rewards[task.task_id]) < ATTEMPTS_PER_TASK)
    if missing:
        raise IncompleteCalibrationError(missing)
    return tuple(Measurement(model_identity, task.task_sha256, tuple(rewards[task.task_id])) for task in bank)


@dataclass(frozen=True)
class CheckpointScore:
    checkpoint_identity: str
    development: tuple[float, float]
    retention: float


@dataclass(frozen=True)
class LoopState:
    parent: CheckpointScore
    working: CheckpointScore
    champion: CheckpointScore
    bank: tuple[QualifiedTask, ...]
    completed_pilots: int = 0
    rounds_without_improvement: int = 0
    stop_reason: StopReason | None = None


@dataclass(frozen=True)
class RoundPlan:
    name: str
    current_checkpoint: str
    champion_checkpoint: str
    task_bank: tuple[QualifiedTask, ...]
    bank_identity: str
    calibration_identity: str
    feedback_labels: tuple[str, ...]
    selected_tasks: tuple[QualifiedTask, ...]
    retained_count: int
    fresh_count: int
    absent_bands: tuple[Difficulty, ...]
    development_identity: str
    retention_identity: str
    feedback_identity: str
    runtime_identity: str
    seed: int
    updates: int
    max_glm_responses: int = MAX_GLM_RESPONSES


@dataclass(frozen=True)
class RoundResult:
    candidate: CheckpointScore
    reload_identity: str
    feedback_identity: str
    optimizer_steps: int
    checkpoint_uri: str = ""


def qualified_bank(tasks: tuple[QualifiedTask, ...]) -> tuple[QualifiedTask, ...]:
    """Remove identical duplicates and reject conflicting identities or missing admission records."""
    by_hash: dict[str, QualifiedTask] = {}
    contracts: set[str] = set()
    ids: set[str] = set()
    sources: set[str] = set()
    for task in tasks:
        if task.task_sha256 in by_hash:
            if by_hash[task.task_sha256] != task:
                raise ValueError("Conflicting task identity")
            continue
        if not task.contract_id or task.contract_id in contracts:
            raise ValueError("Qualified tasks require distinct stable semantic contract IDs")
        contracts.add(task.contract_id)
        if task.task_id in ids or task.source_id in sources:
            raise ValueError("Task IDs and source IDs must be unique")
        if not task.admission_sha256 or not task.task_sha256:
            raise ValueError("A task requires sealed admission and content hashes")
        by_hash[task.task_sha256] = task
        ids.add(task.task_id)
        sources.add(task.source_id)
    return tuple(by_hash.values())


def round_inputs(
    state: LoopState,
    bank: tuple[QualifiedTask, ...],
    *,
    bank_identity: str,
    calibration_identity: str,
    feedback_labels: tuple[str, ...],
    development_identity: str,
    retention_identity: str,
    feedback_identity: str,
    runtime_identity: str,
    seed: int,
) -> dict:
    return {
        "current_checkpoint": state.working.checkpoint_identity,
        "champion_checkpoint": state.champion.checkpoint_identity,
        "task_bank": [asdict(task) for task in bank],
        "bank_identity": bank_identity,
        "calibration_identity": calibration_identity,
        "feedback_labels": list(feedback_labels),
        "development_identity": development_identity,
        "retention_identity": retention_identity,
        "feedback_identity": feedback_identity,
        "runtime_identity": runtime_identity,
        "seed": seed,
        "updates": PILOT_UPDATES,
        "max_glm_responses": MAX_GLM_RESPONSES,
    }


def round_plan(
    state: LoopState,
    fresh: tuple[QualifiedTask, ...],
    measurements: tuple[Measurement, ...],
    *,
    run_id: str,
    bank_identity: str,
    calibration_identity: str,
    feedback_labels: tuple[str, ...],
    development_identity: str,
    retention_identity: str,
    feedback_identity: str,
    runtime_identity: str,
    seed: int,
) -> RoundPlan:
    """Freeze sixteen tasks and target equal retained and fresh groups after pilot one."""
    if state.stop_reason is not None or state.completed_pilots >= MAX_PILOTS:
        raise ValueError("The loop stopped")
    bank = qualified_bank((*state.bank, *fresh))
    if len(bank) < MINIMUM_TASKS:
        raise ValueError("Fewer than sixteen qualified unique tasks")
    expected = {task.task_sha256 for task in bank}
    measured = {measurement.task_sha256: measurement for measurement in measurements}
    if len(measured) != len(measurements) or set(measured) != expected:
        raise ValueError("Calibration must measure each bank task exactly once")
    if any(item.checkpoint_identity != state.working.checkpoint_identity for item in measurements):
        raise ValueError("Calibration used a different checkpoint")
    bands = {key: item.difficulty for key, item in measured.items()}
    if not any(len(set(item.rewards)) > 1 for item in measurements):
        raise ValueError("No sampled task group has reward variation")

    def ranked(tasks: tuple[QualifiedTask, ...]) -> list[QualifiedTask]:
        order = {Difficulty.INTERMEDIATE: 0, Difficulty.EASY: 1, Difficulty.HARD: 2}
        result = sorted(tasks, key=lambda task: (order[bands[task.task_sha256]], task.task_sha256))
        # Keep easy and hard tasks when the pool contains those bands.
        representatives = [
            next((task for task in result if bands[task.task_sha256] == band), None) for band in Difficulty
        ]
        first = [task for task in representatives if task is not None]
        return first + [task for task in result if task not in first]

    if state.completed_pilots == 0:
        selected = tuple(ranked(bank)[:MINIMUM_TASKS])
    else:
        retained_contracts = {task.contract_id for task in state.bank}
        new_tasks = tuple(
            task
            for task in fresh
            if task.contract_id not in retained_contracts
            and task.relation not in {"variant", "replacement", "alias"}
            and set(feedback_labels).intersection(task.capability.split(","))
        )
        if not new_tasks:
            raise ValueError("task_supply_exhausted: no new qualified capability contract")
        initial = ranked(state.bank)[:8] + ranked(new_tasks)[:8]
        selected = tuple((initial + [task for task in ranked(bank) if task not in initial])[:MINIMUM_TASKS])
    absent = tuple(band for band in Difficulty if band not in set(bands.values()))
    retained_contracts = {task.contract_id for task in state.bank}
    fresh_count = sum(
        task.contract_id not in retained_contracts and task.relation not in {"variant", "replacement", "alias"}
        for task in selected
    )
    retained_count = len(selected) - fresh_count
    identities = round_inputs(
        state,
        bank,
        bank_identity=bank_identity,
        calibration_identity=calibration_identity,
        feedback_labels=feedback_labels,
        development_identity=development_identity,
        retention_identity=retention_identity,
        feedback_identity=feedback_identity,
        runtime_identity=runtime_identity,
        seed=seed,
    )
    return RoundPlan(
        name=f"{run_id}-pilot-{state.completed_pilots + 1}",
        selected_tasks=selected,
        retained_count=retained_count,
        fresh_count=fresh_count,
        absent_bands=absent,
        **{**identities, "task_bank": bank, "feedback_labels": feedback_labels},
    )


def advance(state: LoopState, plan: RoundPlan, result: RoundResult) -> LoopState:
    """Continue tied candidates and select a champion only after strict improvement."""
    if plan.current_checkpoint != state.working.checkpoint_identity:
        raise ValueError("Round input does not match the working checkpoint")
    candidate = result.candidate
    retention_ok = candidate.retention >= max(state.working.retention, state.champion.retention)
    working_ok = retention_ok and all(
        new >= previous for new, previous in zip(candidate.development, state.working.development, strict=True)
    )
    improved = (
        retention_ok
        and all(new >= previous for new, previous in zip(candidate.development, state.champion.development, strict=True))
        and any(new > previous for new, previous in zip(candidate.development, state.champion.development, strict=True))
    )
    champion = candidate if improved else state.champion
    working = candidate if working_ok else champion
    count = state.completed_pilots + 1
    stale = 0 if improved else state.rounds_without_improvement + 1
    reason = StopReason.NO_IMPROVEMENT if stale >= 2 else StopReason.PILOT_LIMIT if count >= MAX_PILOTS else None
    return replace(
        state,
        working=working,
        champion=champion,
        bank=plan.task_bank,
        completed_pilots=count,
        rounds_without_improvement=stale,
        stop_reason=reason,
    )


def write_once(path: StoragePath, value: dict) -> None:
    content = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Immutable record differs from the sealed record")
        return
    path.parent.mkdirs()
    path.write_text(content)


def seal_round(
    directory: StoragePath,
    state: LoopState,
    plan: RoundPlan,
    result: RoundResult,
    previous_sha256: str,
    replay_plan: dict | None = None,
) -> str:
    """Write a result to regional storage before the next round starts."""
    payload = {
        "protocol": "bootstrap-and-replay-v1",
        "previous_sha256": previous_sha256,
        "plan": asdict(plan),
        "result": asdict(result),
        "state": asdict(state),
    }
    if replay_plan is not None:
        payload["training_replay_plan"] = replay_plan
    digest = compact_json_sha256(payload)
    write_once(directory / f"{plan.name}.json", {"sha256": digest, "payload": payload})
    return digest


def checkpoint_score(value: dict) -> CheckpointScore:
    return CheckpointScore(value["checkpoint_identity"], tuple(value["development"]), value["retention"])


@dataclass(frozen=True)
class ResumedRound:
    plan: RoundPlan
    state: LoopState
    result: RoundResult
    sha256: str
    replay_plan: dict | None


def load_round(path: StoragePath, expected_inputs: dict, previous_sha256: str) -> ResumedRound:
    """Verify known inputs before calibration and restore the sealed selected dataset."""
    record = json.loads(path.read_text())
    restored = restored_round(record)
    payload = record["payload"]
    expected = json.loads(json.dumps(expected_inputs))
    plan_value = payload["plan"]
    if payload["previous_sha256"] != previous_sha256 or any(plan_value[key] != value for key, value in expected.items()):
        raise ValueError("Round resume input mismatch")
    return restored


def restored_round(record: dict) -> ResumedRound:
    """Restore a round after validation of its sealed payload digest."""
    payload = record["payload"]
    if record["sha256"] != compact_json_sha256(payload):
        raise ValueError("Round manifest digest mismatch")
    plan_value = payload["plan"]
    plan = RoundPlan(
        **{
            **plan_value,
            "task_bank": tuple(QualifiedTask(**task) for task in plan_value["task_bank"]),
            "selected_tasks": tuple(QualifiedTask(**task) for task in plan_value["selected_tasks"]),
            "feedback_labels": tuple(plan_value["feedback_labels"]),
            "absent_bands": tuple(Difficulty(band) for band in plan_value["absent_bands"]),
        }
    )
    value = payload["state"]
    state = LoopState(
        parent=checkpoint_score(value["parent"]),
        working=checkpoint_score(value["working"]),
        champion=checkpoint_score(value["champion"]),
        bank=tuple(QualifiedTask(**task) for task in value["bank"]),
        completed_pilots=value["completed_pilots"],
        rounds_without_improvement=value["rounds_without_improvement"],
        stop_reason=StopReason(value["stop_reason"]) if value["stop_reason"] else None,
    )
    result = RoundResult(**{**payload["result"], "candidate": checkpoint_score(payload["result"]["candidate"])})
    return ResumedRound(plan, state, result, record["sha256"], payload.get("training_replay_plan"))


@dataclass(frozen=True)
class FrozenRoundConfig:
    bank_path: str
    plan: RoundPlan
    output_path: str


def freeze_round_dataset(config: FrozenRoundConfig) -> None:
    """Write exactly sixteen distinct qualified tasks in deterministic order."""
    from taskcompendium.parquet import read_tasks, write_tasks  # noqa: PLC0415

    selected = {task.task_id: task for task in config.plan.selected_tasks}
    if len(selected) != MINIMUM_TASKS:
        raise ValueError("A frozen round requires sixteen unique tasks")
    rows = list(read_tasks(prefix_join(config.bank_path, "train.parquet")))
    by_id = {task.id: task for task in rows}
    if len(by_id) != len(rows):
        raise ValueError("The qualified bank contains duplicate task IDs")
    tasks = []
    for identifier, record in sorted(selected.items()):
        task = by_id[identifier]
        if canonical_sha256(task.model_dump(mode="json")) != record.task_sha256:
            raise ValueError("Selected TaskSpec content differs from its sealed bank")
        admission = StoragePath(
            prefix_join(config.bank_path, f"evidence/{record.admission_sha256}/proposal.json")
        ).read_bytes()
        if hashlib.sha256(admission).hexdigest() != record.admission_sha256:
            raise ValueError("Selected admission report differs from its sealed bank")
        report = json.loads(admission)
        if report["task_sha256"] != record.task_sha256 or report["source_group"] != record.source_id:
            raise ValueError("Selected task differs from its sealed evidence")
        tasks.append(task)
    write_tasks(prefix_join(config.output_path, "train.parquet"), tasks)
    StoragePath(prefix_join(config.output_path, "round-plan.json")).write_text(json.dumps(asdict(config.plan)) + "\n")


@dataclass(frozen=True)
class BankExportConfig:
    audit_path: str
    audit_sha256: str
    selection_path: str
    selection_sha256: str
    feedback_identity: str
    output_path: str
    contract_registry_path: str
    contract_registry_sha256: str


def export_qualified_bank(config: BankExportConfig) -> None:
    """Export a separate semantic selection with actual original admission evidence."""
    from taskcompendium.parquet import read_tasks, write_tasks  # noqa: PLC0415

    audit_bytes = pinned_bytes(config.audit_path, config.audit_sha256)
    selection_bytes = pinned_bytes(config.selection_path, config.selection_sha256)
    audit = json.loads(audit_bytes)
    selection = json.loads(selection_bytes)
    registry_bytes = pinned_bytes(config.contract_registry_path, config.contract_registry_sha256)
    registry = json.loads(registry_bytes)
    if registry["selection_sha256"] != config.selection_sha256:
        raise ValueError("Contract registry refers to a different semantic selection")
    if selection["audit_sha256"] != config.audit_sha256:
        raise ValueError("Semantic selection refers to a different audit")
    keys = selection["selected_proposal_keys"]
    if len(keys) != len(set(keys)):
        raise ValueError("Semantic selection contains duplicate proposal keys")
    permitted = set(audit["summary"]["strong_proposal_keys"]) | {
        item["proposal_key"] for item in selection["decisions"] if item["decision"] == "retain"
    }
    if not set(keys) <= permitted:
        raise ValueError("Unresolved or excluded contracts cannot enter the qualified bank")
    proposals = {item["proposal_key"]: item for item in audit["proposals"]}
    bank, tasks = [], []
    for key in sorted(keys):
        proposal = proposals[key]
        parquet_bytes = pinned_bytes(proposal["parquet_path"], proposal["parquet_sha256"])
        if not parquet_bytes:
            raise ValueError("Qualified source parquet is empty")
        task = next(task for task in read_tasks(proposal["parquet_path"]) if task.id == proposal["task_id"])
        if canonical_sha256(task.model_dump(mode="json")) != proposal["task_sha256"]:
            raise ValueError("Qualified TaskSpec changed after the semantic audit")
        if task.metadata["split"] != "train":
            raise ValueError("Evaluation tasks cannot enter the qualified bank")
        root = proposal["local"] if "local" in proposal else proposal["local_evidence"]
        originals: dict[str, bytes] = {}
        for filename, field in (
            ("snapshot.json", "snapshot_sha256"),
            ("repair.json", "repair_sha256"),
            ("acceptance.json", "acceptance_sha256"),
        ):
            originals[filename] = pinned_bytes(prefix_join(root, filename), proposal[field])
        for filename, expected_hash in proposal.get("evidence_file_hashes", {}).items():
            originals[filename] = pinned_bytes(prefix_join(root, filename), expected_hash)
        for filename, expected_hash in proposal.get("attempt_file_hashes", {}).items():
            originals[filename] = pinned_bytes(prefix_join(root, filename), expected_hash)
        if "recorded_admission_identity" in proposal:
            identity = proposal["recorded_admission_identity"]
            originals["admission-identity.json"] = StoragePath(prefix_join(root, "admission-identity.json")).read_bytes()
            if json.loads(originals["admission-identity.json"]) != identity:
                raise ValueError("Recorded admission identity changed")
            metadata = dict(task.metadata)
            metadata.pop("family")
            before_partition = task.model_copy(update={"metadata": metadata})
            pre_hash = hashlib.sha256(before_partition.model_dump_json().encode()).hexdigest()
            if (
                pre_hash != proposal["pre_partition_task_spec_sha256"]
                or pre_hash != identity["inputs"]["task_spec_sha256"]
            ):
                raise ValueError("Pre-partition admission does not identify the final TaskSpec")
        # Preserve the source audit and original reports. Do not create replacement acceptance flags.
        proof = json.dumps(proposal, sort_keys=True).encode()
        proof_hash = hashlib.sha256(proof).hexdigest()
        proof_root = prefix_join(config.output_path, f"evidence/{proof_hash}")
        StoragePath(prefix_join(proof_root, "proposal.json")).write_bytes(proof)
        for filename, content in originals.items():
            StoragePath(prefix_join(proof_root, filename)).write_bytes(content)
        bank.append(
            QualifiedTask(
                task.id,
                proposal["task_sha256"],
                proof_hash,
                proposal["source_group"],
                ",".join(registry["entries"][key]["capability_labels"]),
                registry["entries"][key]["contract_id"],
                registry["entries"][key]["relation"],
            )
        )
        tasks.append(task)
    qualified = qualified_bank(tuple(bank))
    write_tasks(prefix_join(config.output_path, "train.parquet"), sorted(tasks, key=lambda task: task.id))
    StoragePath(prefix_join(config.output_path, "source-audit.json")).write_bytes(audit_bytes)
    StoragePath(prefix_join(config.output_path, "semantic-selection.json")).write_bytes(selection_bytes)
    StoragePath(prefix_join(config.output_path, "contract-registry.json")).write_bytes(registry_bytes)
    StoragePath(prefix_join(config.output_path, "bank.json")).write_text(
        json.dumps(
            {
                "tasks": [asdict(task) for task in qualified],
                "feedback_identity": config.feedback_identity,
                "audit_sha256": config.audit_sha256,
                "selection_sha256": config.selection_sha256,
                "contract_registry_sha256": config.contract_registry_sha256,
            }
        )
        + "\n"
    )
