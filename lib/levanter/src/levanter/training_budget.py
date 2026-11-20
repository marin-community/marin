# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Stop between updates while reserving time for native finalization."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from time import monotonic, time

import jax
import numpy as np
from jax.experimental.multihost_utils import broadcast_one_to_all
from rigging.filesystem.storage_path import StoragePath, prefix_join

from levanter.checkpoint import load_checkpoint_metadata

COMPLETION_FILE = "completion.json"


class TrainingStopReason(StrEnum):
    MAXIMUM_UPDATES = "maximum_updates"
    TIME_BOUND = "time_bound"
    DATA_EXHAUSTED = "data_exhausted"


@dataclass(frozen=True)
class WallClockCooldown:
    """Linear LR cooldown over the final fraction of usable training time."""

    fraction: float
    min_lr_ratio: float

    def __post_init__(self):
        if not 0 < self.fraction <= 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Cooldown requires a fraction in (0, 1] and an LR ratio in [0, 1]")


@dataclass(frozen=True)
class TrainingBudgetConfig:
    """Execution duration and explicit reserves for updates and final checkpointing."""

    duration: timedelta
    finalization_reserve: timedelta
    update_reserve: timedelta
    deadline_epoch_seconds: float | None = None
    cooldown: WallClockCooldown | None = None

    def __post_init__(self):
        if min(self.finalization_reserve, self.update_reserve) < timedelta(0):
            raise ValueError("Training reserves must be nonnegative")
        if self.duration <= self.finalization_reserve + self.update_reserve:
            raise ValueError("Training duration must exceed its finalization and update reserves")

    @property
    def training_duration(self) -> timedelta:
        return self.duration - self.finalization_reserve - self.update_reserve


def budget_learning_rate_scale(budget: TrainingBudgetConfig, remaining: float) -> float:
    """Return the update multiplier for remaining usable training seconds."""
    if budget.cooldown is None:
        return 1.0
    cooldown = budget.cooldown
    window = cooldown.fraction * budget.training_duration.total_seconds()
    return min(
        1.0, max(cooldown.min_lr_ratio, cooldown.min_lr_ratio + (1 - cooldown.min_lr_ratio) * remaining / window)
    )


class TimeLimitedIterator[T](Iterator[T]):
    """Coordinate a between-update stop using rank zero's monotonic clock."""

    def __init__(self, batches: Iterable[T], budget: TrainingBudgetConfig):
        if budget.deadline_epoch_seconds is None:
            raise ValueError("Resolve the execution deadline before starting time-limited training")
        self.batches = iter(batches)
        self.budget = budget
        self.cutoff = (
            monotonic()
            + budget.deadline_epoch_seconds
            - time()
            - (budget.finalization_reserve + budget.update_reserve).total_seconds()
        )
        self.stop_reason: TrainingStopReason | None = None
        self.learning_rate_scale = 1.0

    def __iter__(self) -> "TimeLimitedIterator[T]":
        return self

    def __next__(self) -> T:
        remaining = self.cutoff - monotonic()
        decision = broadcast_one_to_all(
            np.asarray([remaining <= 0, budget_learning_rate_scale(self.budget, remaining)], dtype=np.float32),
            is_source=jax.process_index() == 0,
        )
        if bool(decision[0]):
            self.stop_reason = TrainingStopReason.TIME_BOUND
            raise StopIteration
        try:
            batch = next(self.batches)
        except StopIteration:
            self.stop_reason = TrainingStopReason.DATA_EXHAUSTED
            raise
        self.learning_rate_scale = float(decision[1])
        return batch


def record_training_completion(
    *,
    telemetry_root: str,
    checkpoint: str,
    completed_updates: int,
    maximum_updates: int,
    budget: TrainingBudgetConfig,
    stop_reason: TrainingStopReason | None,
    final_learning_rate_scale: float,
) -> None:
    """Write a completion receipt only after the final native checkpoint commits."""
    if jax.process_index() != 0:
        return
    reason = TrainingStopReason.MAXIMUM_UPDATES if completed_updates == maximum_updates else stop_reason
    if reason is None or not 0 < completed_updates <= maximum_updates:
        raise ValueError("Training stopped without a valid completed update count or stop reason")
    metadata = load_checkpoint_metadata(checkpoint)
    if metadata["step"] != completed_updates - 1 or metadata["is_temporary"]:
        raise ValueError("Time-limited training has no complete final permanent checkpoint")
    finalized = time()
    deadline = budget.deadline_epoch_seconds
    assert deadline is not None
    if finalized > deadline:
        raise RuntimeError("Training finalization exceeded its execution deadline")
    receipt = {
        "schema": 1,
        "completed_updates": completed_updates,
        "maximum_updates": maximum_updates,
        "checkpoint": checkpoint,
        "step": completed_updates - 1,
        "stop_reason": reason,
        "duration_seconds": budget.duration.total_seconds(),
        "deadline_epoch_seconds": budget.deadline_epoch_seconds,
        "finalization_reserve_seconds": budget.finalization_reserve.total_seconds(),
        "update_reserve_seconds": budget.update_reserve.total_seconds(),
        "finalized_epoch_seconds": finalized,
        "finalized_at": datetime.fromtimestamp(finalized, UTC).isoformat(),
        "cooldown": asdict(budget.cooldown) if budget.cooldown is not None else None,
        "final_learning_rate_scale": final_learning_rate_scale,
    }
    (StoragePath(telemetry_root) / COMPLETION_FILE).write_text(json.dumps(receipt, indent=2))


def read_training_completion(
    *, telemetry_root: str, checkpoint_root: str, maximum_updates: int, budget: TrainingBudgetConfig
) -> dict:
    """Verify the actual update count, stop reason, deadline and committed checkpoint."""
    receipt = json.loads((StoragePath(telemetry_root) / COMPLETION_FILE).read_text())
    completed = receipt["completed_updates"]
    reason = TrainingStopReason(receipt["stop_reason"])
    if receipt["maximum_updates"] != maximum_updates or not 0 < completed <= maximum_updates:
        raise ValueError("Training completion update count differs from the configured maximum")
    if (reason == TrainingStopReason.MAXIMUM_UPDATES) != (completed == maximum_updates):
        raise ValueError("Training completion stop reason disagrees with its update count")
    if reason == TrainingStopReason.DATA_EXHAUSTED:
        raise ValueError("Training exhausted its data before the controlled duration or update maximum")
    if budget.cooldown is not None:
        if receipt["cooldown"] != asdict(budget.cooldown):
            raise ValueError("Training completion changed the declared wall-clock cooldown")
        if reason == TrainingStopReason.TIME_BOUND:
            floor = budget.cooldown.min_lr_ratio
            if not abs(receipt["final_learning_rate_scale"] - floor) <= max(floor * 0.1, 1e-7):
                raise ValueError("Training final-update LR scale is not within 10% of the declared minimum")
    for key, expected in (
        ("duration_seconds", budget.duration.total_seconds()),
        ("finalization_reserve_seconds", budget.finalization_reserve.total_seconds()),
        ("update_reserve_seconds", budget.update_reserve.total_seconds()),
    ):
        if receipt[key] != expected:
            raise ValueError(f"Training completion changed {key}")
    if receipt["finalized_epoch_seconds"] > receipt["deadline_epoch_seconds"]:
        raise ValueError("Training completed after its execution deadline")
    expected_checkpoint = prefix_join(checkpoint_root, f"step-{completed - 1}")
    if receipt["checkpoint"] != expected_checkpoint or receipt["step"] != completed - 1:
        raise ValueError("Training completion identifies a different final checkpoint")
    metadata = load_checkpoint_metadata(expected_checkpoint)
    if metadata["step"] != completed - 1 or metadata["is_temporary"]:
        raise ValueError("Training completion checkpoint is not permanently committed")
    return receipt
