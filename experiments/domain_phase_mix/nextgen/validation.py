# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation planning, slot execution, and state finalization."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

import fsspec
import pandas as pd

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

from experiments.domain_phase_mix.nextgen.contracts import (
    Candidate,
    LoopConfig,
    LoopState,
    PolicyArtifactRef,
    RunRecord,
    ValidationRecord,
)
from experiments.domain_phase_mix.nextgen.fit_propose import CANDIDATE_ASSIGNMENTS_JSON, CANDIDATES_JSON
from experiments.domain_phase_mix.nextgen.state_store import STATE_FILE, load_loop_state, write_loop_state
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

logger = logging.getLogger(__name__)
_VALIDATION_EXECUTION_ADAPTER: ValidationExecutionAdapter | None = None

VALIDATION_PLAN_JSON = "validation_plan.json"
PENDING_CANDIDATES_JSON = "pending_candidates.json"
SLOT_ASSIGNMENTS_JSON = "slot_assignments.json"
SLOT_RESULT_JSON = "slot_result.json"
VALIDATION_RESULTS_JSON = "validation_results.json"
VALIDATION_RESULTS_CSV = "validation_results.csv"


class ValidationExecutionAdapter(Protocol):
    """Adapter interface for executing validation candidates."""

    def execute(self, *, model_name: str, candidate: Candidate, output_path: str) -> ValidationRecord:
        ...


def register_validation_execution_adapter(adapter: ValidationExecutionAdapter | None) -> None:
    """Register or clear the runtime validation execution adapter."""
    global _VALIDATION_EXECUTION_ADAPTER
    _VALIDATION_EXECUTION_ADAPTER = adapter


@dataclass(frozen=True)
class PlanValidationConfig:
    output_path: str
    fit_output_path: InputName | str
    state_output_path: InputName | str
    model_names_json: str


@dataclass(frozen=True)
class ValidationSlotConfig:
    output_path: str
    plan_output_path: InputName | str
    fit_output_path: InputName | str
    model_name: str
    execute_slot: bool = False


@dataclass(frozen=True)
class CollectValidationConfig:
    output_path: str
    slot_output_paths: tuple[InputName, ...]
    slot_model_names_json: str


@dataclass(frozen=True)
class FinalizeStateConfig:
    output_path: str
    state_output_path: InputName | str
    validation_output_path: InputName | str
    merge_output_path: InputName | str


def _load_json(path: str, default):
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return default
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _decode_candidate(row: dict) -> Candidate:
    policy_ref_row = row.get("policy_ref")
    policy_ref = None
    if isinstance(policy_ref_row, dict):
        policy_ref = PolicyArtifactRef(
            uri=policy_ref_row["uri"],
            format=policy_ref_row.get("format", "json"),
        )
    return Candidate(
        candidate_id=row["candidate_id"],
        model_name=row["model_name"],
        kind=row["kind"],
        phase_weights=row.get("phase_weights"),
        policy_ref=policy_ref,
        predicted_objective=float(row["predicted_objective"]),
    )


def _decode_run_record(row: dict):
    return RunRecord(
        wandb_run_id=row.get("wandb_run_id"),
        source_experiment=row["source_experiment"],
        local_run_id=row.get("local_run_id"),
        run_name=row.get("run_name"),
        phase_weights=normalize_phase_weights(row.get("phase_weights", {})),
        status=row.get("status", "unknown"),
        metrics={str(k): float(v) for k, v in row.get("metrics", {}).items()},
    )


def plan_validation(config: PlanValidationConfig) -> None:
    """Plan pending validation candidates and static slot assignments."""
    fit_dir = str(config.fit_output_path)
    state_dir = str(config.state_output_path)
    model_names = tuple(json.loads(config.model_names_json))

    candidate_rows = _load_json(os.path.join(fit_dir, CANDIDATES_JSON), default=[])
    assignment = _load_json(os.path.join(fit_dir, CANDIDATE_ASSIGNMENTS_JSON), default={})
    candidates = {row["candidate_id"]: _decode_candidate(row) for row in candidate_rows}

    state = load_loop_state(os.path.join(state_dir, STATE_FILE))

    pending_ids: set[str] = set()
    plan_rows: list[ValidationRecord] = []

    for model_name in model_names:
        candidate_id = assignment.get(model_name)
        if candidate_id is None:
            plan_rows.append(
                ValidationRecord(
                    candidate_id=f"none:{model_name}",
                    model_name=model_name,
                    status="skipped",
                    details={"reason": "no_candidate_assignment"},
                )
            )
            continue

        existing = state.validated_candidates.get(candidate_id)
        if existing is not None:
            plan_rows.append(
                ValidationRecord(
                    candidate_id=candidate_id,
                    model_name=model_name,
                    status="reused",
                    wandb_run_id=existing.wandb_run_id,
                    metric_value=existing.metric_value,
                    details={"reason": "already_validated", "existing_status": existing.status},
                )
            )
            continue

        pending_ids.add(candidate_id)
        plan_rows.append(
            ValidationRecord(
                candidate_id=candidate_id,
                model_name=model_name,
                status="pending",
            )
        )

    pending = [dataclasses.asdict(candidates[cid]) for cid in sorted(pending_ids) if cid in candidates]

    slot_assignments: dict[str, str | None] = {}
    for model_name in model_names:
        candidate_id = assignment.get(model_name)
        if candidate_id in pending_ids:
            slot_assignments[model_name] = candidate_id
        else:
            slot_assignments[model_name] = None

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, VALIDATION_PLAN_JSON), "w") as f:
        json.dump([dataclasses.asdict(row) for row in plan_rows], f, indent=2, sort_keys=True)

    with fsspec.open(os.path.join(config.output_path, PENDING_CANDIDATES_JSON), "w") as f:
        json.dump(pending, f, indent=2, sort_keys=True)

    with fsspec.open(os.path.join(config.output_path, SLOT_ASSIGNMENTS_JSON), "w") as f:
        json.dump(slot_assignments, f, indent=2, sort_keys=True)



def run_validation_slot(config: ValidationSlotConfig) -> None:
    """Static slot that executes or records candidate validation handling."""
    plan_dir = str(config.plan_output_path)
    fit_dir = str(config.fit_output_path)

    assignments = _load_json(os.path.join(plan_dir, SLOT_ASSIGNMENTS_JSON), default={})
    candidate_rows = _load_json(os.path.join(fit_dir, CANDIDATES_JSON), default=[])
    candidate_by_id = {row["candidate_id"]: _decode_candidate(row) for row in candidate_rows}

    candidate_id = assignments.get(config.model_name)
    if candidate_id is None:
        result = ValidationRecord(
            candidate_id=f"none:{config.model_name}",
            model_name=config.model_name,
            status="skipped",
            details={"reason": "no_pending_candidate"},
        )
    else:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            result = ValidationRecord(
                candidate_id=candidate_id,
                model_name=config.model_name,
                status="failed",
                details={"reason": "candidate_missing_from_fit_output"},
            )
        elif not config.execute_slot:
            result = ValidationRecord(
                candidate_id=candidate_id,
                model_name=config.model_name,
                status="planned",
                metric_value=candidate.predicted_objective,
                details={"reason": "validation_execution_disabled"},
            )
        elif _VALIDATION_EXECUTION_ADAPTER is not None:
            try:
                result = _VALIDATION_EXECUTION_ADAPTER.execute(
                    model_name=config.model_name,
                    candidate=candidate,
                    output_path=config.output_path,
                )
            except Exception as exc:
                logger.exception("Validation adapter execution failed for model %s", config.model_name)
                result = ValidationRecord(
                    candidate_id=candidate_id,
                    model_name=config.model_name,
                    status="failed",
                    details={"reason": "validation_execution_error", "error": str(exc)},
                )
        else:
            result = ValidationRecord(
                candidate_id=candidate_id,
                model_name=config.model_name,
                status="failed",
                details={"reason": "validation_execution_adapter_not_registered"},
            )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, SLOT_RESULT_JSON), "w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2, sort_keys=True)



def collect_validation_results(config: CollectValidationConfig) -> None:
    """Collect slot outputs into one validation results artifact."""
    slot_model_names = json.loads(config.slot_model_names_json)
    slot_paths = list(config.slot_output_paths)

    rows: list[dict] = []
    for model_name, slot_path in zip(slot_model_names, slot_paths, strict=True):
        result_path = os.path.join(slot_path, SLOT_RESULT_JSON)
        result = _load_json(result_path, default=None)
        if result is None:
            rows.append(
                dataclasses.asdict(
                    ValidationRecord(
                        candidate_id=f"none:{model_name}",
                        model_name=model_name,
                        status="failed",
                        details={"reason": "missing_slot_result"},
                    )
                )
            )
            continue

        rows.append(result)

    df = pd.DataFrame(rows)
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, VALIDATION_RESULTS_JSON), "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    with fsspec.open(os.path.join(config.output_path, VALIDATION_RESULTS_CSV), "w") as f:
        df.to_csv(f, index=False)



def finalize_state(config: FinalizeStateConfig) -> None:
    """Update persistent state with validation outcomes and run registry snapshot."""
    state_dir = str(config.state_output_path)
    validation_dir = str(config.validation_output_path)

    state = load_loop_state(os.path.join(state_dir, STATE_FILE))
    merged_snapshot = _load_json(os.path.join(str(config.merge_output_path), "state_snapshot.json"), default=None)
    if merged_snapshot is not None:
        state = LoopState(
            loop_name=merged_snapshot["loop_name"],
            objective_metric=merged_snapshot["objective_metric"],
            next_local_run_id=int(merged_snapshot.get("next_local_run_id", state.next_local_run_id)),
            runs=state.runs if not merged_snapshot.get("runs") else [
                _decode_run_record(row) for row in merged_snapshot.get("runs", [])
            ],
            validated_candidates=state.validated_candidates,
        )
    validation_rows = _load_json(os.path.join(validation_dir, VALIDATION_RESULTS_JSON), default=[])

    updated_validated = dict(state.validated_candidates)
    for row in validation_rows:
        record = ValidationRecord(
            candidate_id=row["candidate_id"],
            model_name=row["model_name"],
            status=row["status"],
            wandb_run_id=row.get("wandb_run_id"),
            metric_value=row.get("metric_value"),
            details=row.get("details", {}),
        )
        # Treat planned as sufficient for dedupe in subsequent submissions.
        if record.candidate_id.startswith("none:"):
            continue
        if record.status in {"planned", "completed", "reused", "skipped"}:
            updated_validated[record.candidate_id] = record

    next_state = dataclasses.replace(state, validated_candidates=updated_validated)
    write_loop_state(os.path.join(state_dir, STATE_FILE), next_state)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, "finalized_state_summary.json"), "w") as f:
        json.dump(
            {
                "loop_name": next_state.loop_name,
                "n_runs": len(next_state.runs),
                "next_local_run_id": next_state.next_local_run_id,
                "n_validated_candidates": len(next_state.validated_candidates),
            },
            f,
            indent=2,
            sort_keys=True,
        )



def create_plan_validation_step(
    *,
    loop: LoopConfig,
    fit_step: ExecutorStep,
    state_step: ExecutorStep,
    output_override_path: str,
) -> ExecutorStep:
    """Create plan/validation step."""
    return ExecutorStep(
        name=f"{loop.name}/plan/validation",
        description="Plan pending candidate validations and slot assignments",
        fn=plan_validation,
        config=PlanValidationConfig(
            output_path=this_output_path(),
            fit_output_path=output_path_of(fit_step),
            state_output_path=output_path_of(state_step),
            model_names_json=json.dumps(list(loop.model_names), sort_keys=True),
        ),
        override_output_path=output_override_path,
    )


def create_validation_slot_step(
    *,
    loop: LoopConfig,
    model_name: str,
    fit_step: ExecutorStep,
    plan_step: ExecutorStep,
    output_override_path: str,
) -> ExecutorStep:
    """Create a static candidate slot step for one model name."""
    return ExecutorStep(
        name=f"{loop.name}/validate/{model_name}",
        description=f"Validation slot for model '{model_name}'",
        fn=run_validation_slot,
        config=ValidationSlotConfig(
            output_path=this_output_path(),
            plan_output_path=output_path_of(plan_step),
            fit_output_path=output_path_of(fit_step),
            model_name=model_name,
            execute_slot=loop.execute_validation_slots,
        ),
        override_output_path=output_override_path,
    )


def create_collect_validation_step(
    *,
    loop_name: str,
    slot_steps: list[ExecutorStep],
    output_override_path: str,
) -> ExecutorStep:
    """Create collect/validation_results step."""
    slot_model_names = [step.name.split("/")[-1] for step in slot_steps]

    return ExecutorStep(
        name=f"{loop_name}/collect/validation_results",
        description="Aggregate static validation slot outputs",
        fn=collect_validation_results,
        config=CollectValidationConfig(
            output_path=this_output_path(),
            slot_output_paths=tuple(output_path_of(step) for step in slot_steps),
            slot_model_names_json=json.dumps(slot_model_names, sort_keys=True),
        ),
        override_output_path=output_override_path,
    )


def create_finalize_state_step(
    *,
    loop_name: str,
    state_step: ExecutorStep,
    validation_collect_step: ExecutorStep,
    merge_step: ExecutorStep,
    output_override_path: str,
) -> ExecutorStep:
    """Create final state update step."""
    return ExecutorStep(
        name=f"{loop_name}/report/finalize_state",
        description="Update persistent state with validation outcomes",
        fn=finalize_state,
        config=FinalizeStateConfig(
            output_path=this_output_path(),
            state_output_path=output_path_of(state_step),
            validation_output_path=output_path_of(validation_collect_step),
            merge_output_path=output_path_of(merge_step),
        ),
        override_output_path=output_override_path,
    )
