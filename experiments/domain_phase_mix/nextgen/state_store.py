# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent state helpers and executor step for next-gen loops."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from marin.execution.executor import ExecutorStep, this_output_path

from experiments.domain_phase_mix.nextgen.contracts import (
    LoopConfig,
    LoopState,
    RunRecord,
    ValidationRecord,
)
from experiments.domain_phase_mix.nextgen.utils import read_json, write_json

STATE_FILE = "loop_state.json"


@dataclass(frozen=True)
class BootstrapStateConfig:
    """Executor config for state bootstrap/load."""

    output_path: str
    loop_name: str
    objective_metric: str


def _state_path(output_path: str) -> str:
    return os.path.join(output_path, STATE_FILE)


def _decode_run_records(rows: list[dict]) -> list[RunRecord]:
    return [
        RunRecord(
            wandb_run_id=row.get("wandb_run_id"),
            source_experiment=row["source_experiment"],
            local_run_id=row.get("local_run_id"),
            run_name=row.get("run_name"),
            phase_weights=row.get("phase_weights", {}),
            status=row.get("status", "unknown"),
            metrics=row.get("metrics", {}),
        )
        for row in rows
    ]


def _decode_validated(rows: dict[str, dict]) -> dict[str, ValidationRecord]:
    return {
        cid: ValidationRecord(
            candidate_id=payload["candidate_id"],
            model_name=payload["model_name"],
            status=payload["status"],
            wandb_run_id=payload.get("wandb_run_id"),
            metric_value=payload.get("metric_value"),
            details=payload.get("details", {}),
        )
        for cid, payload in rows.items()
    }


def load_loop_state(path: str, *, loop_name: str | None = None, objective_metric: str | None = None) -> LoopState:
    """Load state from *path*; return a new default state if missing."""
    state_payload = read_json(path, default=None)
    if state_payload is None:
        if loop_name is None or objective_metric is None:
            raise ValueError("loop_name/objective_metric required when state does not exist")
        return LoopState(loop_name=loop_name, objective_metric=objective_metric)

    return LoopState(
        loop_name=state_payload["loop_name"],
        objective_metric=state_payload["objective_metric"],
        next_local_run_id=int(state_payload.get("next_local_run_id", 0)),
        runs=_decode_run_records(state_payload.get("runs", [])),
        validated_candidates=_decode_validated(state_payload.get("validated_candidates", {})),
    )


def write_loop_state(path: str, state: LoopState) -> None:
    """Write loop state JSON."""
    write_json(path, asdict(state))


def bootstrap_or_load_state(config: BootstrapStateConfig) -> None:
    """Bootstrap persistent loop state in the target output path."""
    path = _state_path(config.output_path)
    state = load_loop_state(path, loop_name=config.loop_name, objective_metric=config.objective_metric)
    write_loop_state(path, state)


def create_bootstrap_step(loop: LoopConfig, output_override_path: str) -> ExecutorStep:
    """Create state bootstrap step with stable output path."""
    return ExecutorStep(
        name=f"{loop.name}/state/bootstrap_or_load",
        description="Load persistent loop state or initialize defaults",
        fn=bootstrap_or_load_state,
        config=BootstrapStateConfig(
            output_path=this_output_path(),
            loop_name=loop.name,
            objective_metric=loop.objective_metric,
        ),
        override_output_path=output_override_path,
    )
