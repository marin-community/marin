# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the next-generation domain/phase mixture optimization loop.

These contracts are intentionally filesystem- and executor-friendly so we can
persist loop state across submissions and re-run only downstream stages when
new runs or models are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class ImportSource(Protocol):
    """Interface for trajectory/run ingestion sources."""

    def collect_runs(self) -> list[RunRecord]:
        """Collect run-level metadata records."""
        ...

    def collect_trajectories(self, objective_metric: str):
        """Collect trajectory rows for *objective_metric*.

        Implementations should return a pandas DataFrame, but the type is left
        unconstrained here to avoid importing pandas in this contract module.
        """
        ...


@dataclass(frozen=True)
class PolicyArtifactRef:
    """Reference to an externally-trained policy artifact.

    For v1, policy artifacts are expected to be JSON files that can be resolved
    through fsspec and include enough information to produce per-phase actions.
    """

    uri: str
    format: str = "json"


@dataclass(frozen=True)
class LoopConfig:
    """Configuration for a persisted next-gen mixture loop."""

    name: str
    objective_metric: str
    model_names: tuple[str, ...]
    n_new_runs: int = 0
    import_sources: tuple[ImportSource, ...] = ()
    validation_policy: Literal["top1_per_model_dedup"] = "top1_per_model_dedup"
    trajectory_granularity: Literal["eval_checkpoints_only"] = "eval_checkpoints_only"
    state_root: str = "domain_phase_mix/nextgen"
    # Optional model registry controls
    candidate_search_points: int = 8192
    candidate_search_seed: int = 42
    # Optional validation execution toggle. V1 defaults to planning-only slots;
    # real execution can be enabled when an execution adapter is provided.
    execute_validation_slots: bool = False


@dataclass(frozen=True)
class RunRecord:
    """Canonical run record tracked by the loop state."""

    wandb_run_id: str | None
    source_experiment: str
    local_run_id: int | None
    run_name: str | None
    phase_weights: dict[str, dict[str, float]]
    status: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryPoint:
    """One trajectory row at eval-checkpoint granularity."""

    wandb_run_id: str | None
    source_experiment: str
    local_run_id: int | None
    run_name: str | None
    step: int
    total_tokens: float | None
    metric_key: str
    metric_value: float


@dataclass(frozen=True)
class Candidate:
    """Candidate configuration proposed by a model."""

    candidate_id: str
    model_name: str
    kind: Literal["schedule", "policy"]
    phase_weights: dict[str, dict[str, float]] | None
    policy_ref: PolicyArtifactRef | None
    predicted_objective: float


@dataclass(frozen=True)
class ValidationRecord:
    """Status of a candidate validation attempt."""

    candidate_id: str
    model_name: str
    status: Literal["pending", "reused", "planned", "completed", "skipped", "failed"]
    wandb_run_id: str | None = None
    metric_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedRun:
    """One newly-planned swarm run."""

    local_run_id: int
    run_name: str
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LoopState:
    """Persistent loop state snapshot."""

    loop_name: str
    objective_metric: str
    next_local_run_id: int = 0
    runs: list[RunRecord] = field(default_factory=list)
    validated_candidates: dict[str, ValidationRecord] = field(default_factory=dict)

    def run_by_wandb_id(self) -> dict[str, RunRecord]:
        """Index run records that have a W&B run id."""
        return {r.wandb_run_id: r for r in self.runs if r.wandb_run_id is not None}
