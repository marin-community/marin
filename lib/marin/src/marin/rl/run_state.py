# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL run lifecycle actor.

Tracks whether the training run is running, completed, or failed.
Rollout workers poll this to know when to shut down. The trainer
signals completion or failure and publishes the latest completed
training step. This is separate from weight transfer coordination;
lifecycle is its own concern.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

import fsspec
from marin.rl.rollout_schedule import RolloutAssignment, RolloutScheduleCursor, rollout_assignment
from rigging.filesystem import filesystem as marin_filesystem

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


@dataclass(frozen=True)
class RolloutAssignmentCounters:
    """Cumulative finite-data schedule counters for one worker and lesson."""

    reserved: int = 0
    reused_pending: int = 0
    committed: int = 0
    ledger_recovered: int = 0


@dataclass(frozen=True)
class RolloutAssignmentCommit:
    """Durable commit record for one finite-data rollout assignment."""

    assignment_id: str
    worker_index: int
    lesson_id: str
    worker_seed: int
    dataset_len: int
    epoch: int
    start_position: int
    end_position: int
    n_examples: int


class RolloutScheduleLedger:
    """Small file-backed ledger of committed finite-data assignments."""

    def __init__(self, path: str):
        self.path = path.rstrip("/")
        storage_options = fsspec.utils.infer_storage_options(path)  # type: ignore[attr-defined]
        self.fs = marin_filesystem(storage_options["protocol"] or "file")
        self.fs.makedirs(self.path, exist_ok=True)

    def write_commit(self, assignment: RolloutAssignment) -> None:
        """Persist a committed assignment record idempotently."""
        record = RolloutAssignmentCommit(
            assignment_id=assignment.assignment_id,
            worker_index=assignment.worker_index,
            lesson_id=assignment.lesson_id,
            worker_seed=assignment.worker_seed,
            dataset_len=assignment.dataset_len,
            epoch=assignment.epoch,
            start_position=assignment.start_position,
            end_position=assignment.end_position,
            n_examples=len(assignment.indices),
        )
        final_path = self._commit_path(assignment.assignment_id)
        if self.fs.exists(final_path):
            return
        tmp_path = f"{final_path}.tmp-{os.getpid()}-{time.time_ns()}"
        with self.fs.open(tmp_path, "w") as f:
            json.dump(asdict(record), f, sort_keys=True)
            f.write("\n")
        self.fs.move(tmp_path, final_path)

    def read_commits(self) -> list[RolloutAssignmentCommit]:
        """Read all durable assignment commits in deterministic filename order."""
        commits: list[RolloutAssignmentCommit] = []
        for path in sorted(self.fs.glob(f"{self.path}/*.json")):
            with self.fs.open(path) as f:
                payload = json.load(f)
            commits.append(
                RolloutAssignmentCommit(
                    assignment_id=str(payload["assignment_id"]),
                    worker_index=int(payload["worker_index"]),
                    lesson_id=str(payload["lesson_id"]),
                    worker_seed=int(payload["worker_seed"]),
                    dataset_len=int(payload["dataset_len"]),
                    epoch=int(payload["epoch"]),
                    start_position=int(payload["start_position"]),
                    end_position=int(payload["end_position"]),
                    n_examples=int(payload["n_examples"]),
                )
            )
        return commits

    def _commit_path(self, assignment_id: str) -> str:
        assignment_hash = hashlib.sha256(assignment_id.encode("utf-8")).hexdigest()
        return f"{self.path}/{assignment_hash}.json"


class RLRunState:
    """Lightweight actor that tracks RL run lifecycle.

    Created by the coordinator, shared by trainer and rollout workers.
    """

    def __init__(self, schedule_ledger_path: str | None = None):
        self._status: RunStatus = RunStatus.RUNNING
        self._failure_message: str | None = None
        self._train_step: int = -1
        self._rollout_transfer_counters: dict[int, RolloutTransferCounters] = {}
        self._rollout_schedule_positions: dict[tuple[int, str], int] = {}
        self._pending_rollout_assignments: dict[tuple[int, str], RolloutAssignment] = {}
        self._rollout_assignment_counters: dict[tuple[int, str], RolloutAssignmentCounters] = {}
        self._schedule_ledger = RolloutScheduleLedger(schedule_ledger_path) if schedule_ledger_path else None
        self._recover_rollout_schedule_positions()

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

    def get_rollout_schedule_cursor(self, worker_index: int, lesson_id: str) -> RolloutScheduleCursor:
        """Return the committed finite-data schedule cursor for a logical rollout worker."""
        key = (worker_index, lesson_id)
        return RolloutScheduleCursor(
            worker_index=worker_index,
            lesson_id=lesson_id,
            position=self._rollout_schedule_positions.get(key, 0),
        )

    def reserve_rollout_assignment(
        self,
        worker_index: int,
        lesson_id: str,
        worker_seed: int,
        dataset_len: int,
        n_examples: int,
    ) -> RolloutAssignment:
        """Reserve the next finite-data assignment for a logical rollout worker.

        Reservation is idempotent while an assignment is pending. This lets a
        worker retry the same examples after a process crash before durable
        rollout write or before commit.
        """
        key = (worker_index, lesson_id)
        pending = self._pending_rollout_assignments.get(key)
        if pending is not None:
            if pending.worker_seed != worker_seed:
                raise ValueError(
                    f"pending rollout assignment for worker {worker_index} lesson {lesson_id} "
                    f"uses seed {pending.worker_seed}, got {worker_seed}"
                )
            if pending.dataset_len != dataset_len:
                raise ValueError(
                    f"pending rollout assignment for worker {worker_index} lesson {lesson_id} "
                    f"uses dataset_len {pending.dataset_len}, got {dataset_len}"
                )
            self._add_rollout_assignment_counters(key, reused_pending=1)
            return pending

        position = self._rollout_schedule_positions.get(key, 0)
        assignment = rollout_assignment(
            worker_index=worker_index,
            lesson_id=lesson_id,
            worker_seed=worker_seed,
            dataset_len=dataset_len,
            start_position=position,
            n_examples=n_examples,
        )
        self._pending_rollout_assignments[key] = assignment
        self._add_rollout_assignment_counters(key, reserved=1)
        return assignment

    def commit_rollout_assignment(self, worker_index: int, lesson_id: str, assignment_id: str) -> RolloutScheduleCursor:
        """Commit a previously reserved finite-data assignment after durable rollout write."""
        key = (worker_index, lesson_id)
        pending = self._pending_rollout_assignments.get(key)
        if pending is None:
            raise ValueError(f"no pending rollout assignment for worker {worker_index} lesson {lesson_id}")
        if pending.assignment_id != assignment_id:
            raise ValueError(f"cannot commit assignment {assignment_id}; pending assignment is {pending.assignment_id}")

        if self._schedule_ledger is not None:
            self._schedule_ledger.write_commit(pending)
        self._rollout_schedule_positions[key] = pending.end_position
        del self._pending_rollout_assignments[key]
        self._add_rollout_assignment_counters(key, committed=1)
        return self.get_rollout_schedule_cursor(worker_index, lesson_id)

    def get_rollout_schedule_stats(self) -> dict[str, int]:
        """Return numeric finite-data schedule counters for monitoring."""
        reserved = sum(counters.reserved for counters in self._rollout_assignment_counters.values())
        reused_pending = sum(counters.reused_pending for counters in self._rollout_assignment_counters.values())
        committed = sum(counters.committed for counters in self._rollout_assignment_counters.values())
        ledger_recovered = sum(counters.ledger_recovered for counters in self._rollout_assignment_counters.values())
        return {
            "active_cursors": len(self._rollout_schedule_positions),
            "pending_assignments": len(self._pending_rollout_assignments),
            "reserved_assignments": reserved,
            "reused_pending_assignments": reused_pending,
            "committed_assignments": committed,
            "ledger_recovered_assignments": ledger_recovered,
        }

    def _recover_rollout_schedule_positions(self) -> None:
        if self._schedule_ledger is None:
            return

        recovered_records = 0
        for commit in self._schedule_ledger.read_commits():
            key = (commit.worker_index, commit.lesson_id)
            previous_position = self._rollout_schedule_positions.get(key, 0)
            self._rollout_schedule_positions[key] = max(previous_position, commit.end_position)
            self._add_rollout_assignment_counters(key, ledger_recovered=1)
            recovered_records += 1

        if recovered_records > 0:
            logger.info(
                "Recovered %d finite-data rollout schedule commits across %d cursors",
                recovered_records,
                len(self._rollout_schedule_positions),
            )

    def _add_rollout_assignment_counters(
        self,
        key: tuple[int, str],
        *,
        reserved: int = 0,
        reused_pending: int = 0,
        committed: int = 0,
        ledger_recovered: int = 0,
    ) -> RolloutAssignmentCounters:
        deltas = (reserved, reused_pending, committed, ledger_recovered)
        if any(delta < 0 for delta in deltas):
            raise ValueError(f"rollout assignment counter deltas must be non-negative, got {deltas}")

        previous = self._rollout_assignment_counters.get(key, RolloutAssignmentCounters())
        updated = replace(
            previous,
            reserved=previous.reserved + reserved,
            reused_pending=previous.reused_pending + reused_pending,
            committed=previous.committed + committed,
            ledger_recovered=previous.ledger_recovered + ledger_recovered,
        )
        self._rollout_assignment_counters[key] = updated
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
