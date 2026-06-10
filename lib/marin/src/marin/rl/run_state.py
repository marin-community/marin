# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL run lifecycle actor.

Tracks whether the training run is running, completed, or failed.
Rollout workers poll this to know when to shut down. The trainer
signals completion or failure and publishes the latest completed
training step. This is separate from weight transfer coordination;
lifecycle is its own concern.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunStateSnapshot:
    """Current RL run status and latest completed trainer step."""

    status: str
    train_step: int
    failure_message: str | None


@dataclass(frozen=True)
class RolloutTransferCounters:
    """Resume-safe cumulative transfer counters for a rollout worker."""

    total_polls: int = 0
    successful_receives: int = 0
    failed_receives: int = 0


class RLRunState:
    """Lightweight actor that tracks RL run lifecycle.

    Created by the coordinator, shared by trainer and rollout workers.
    """

    def __init__(self):
        self._status: RunStatus = RunStatus.RUNNING
        self._failure_message: str | None = None
        self._train_step: int = -1
        self._rollout_transfer_counters: dict[int, RolloutTransferCounters] = {}

    def get_status(self) -> str:
        return self._status.value

    def get_snapshot(self) -> RunStateSnapshot:
        return RunStateSnapshot(
            status=self._status.value,
            train_step=self._train_step,
            failure_message=self._failure_message,
        )

    def is_terminal(self) -> bool:
        return self._status != RunStatus.RUNNING

    def update_train_step(self, step: int) -> None:
        if step > self._train_step:
            self._train_step = step

    def get_train_step(self) -> int:
        return self._train_step

    def get_rollout_transfer_counters(self, worker_index: int) -> RolloutTransferCounters:
        return self._rollout_transfer_counters.get(worker_index, RolloutTransferCounters())

    def add_rollout_transfer_counters(
        self,
        worker_index: int,
        total_polls_delta: int,
        successful_receives_delta: int,
        failed_receives_delta: int,
    ) -> RolloutTransferCounters:
        deltas = (total_polls_delta, successful_receives_delta, failed_receives_delta)
        if any(delta < 0 for delta in deltas):
            raise ValueError(f"rollout transfer deltas must be non-negative, got {deltas}")

        previous = self.get_rollout_transfer_counters(worker_index)
        updated = RolloutTransferCounters(
            total_polls=previous.total_polls + total_polls_delta,
            successful_receives=previous.successful_receives + successful_receives_delta,
            failed_receives=previous.failed_receives + failed_receives_delta,
        )
        self._rollout_transfer_counters[worker_index] = updated
        return updated

    def mark_completed(self) -> None:
        if self._status == RunStatus.RUNNING:
            self._status = RunStatus.COMPLETED
            logger.info("RL run marked as completed")

    def mark_failed(self, message: str = "") -> None:
        if self._status == RunStatus.RUNNING:
            self._status = RunStatus.FAILED
            self._failure_message = message
            logger.info("RL run marked as failed: %s", message)

    def get_failure_message(self) -> str | None:
        return self._failure_message
