# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task representation for the controller with state transition logic and retry tracking.

A Task is the unit of execution: it runs on a single worker, has its own lifecycle,
and reports its own state. Jobs with replicas=N expand into N tasks.
"""

from dataclasses import dataclass, field
from enum import Enum

from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2


class TaskTransitionResult(Enum):
    """Result of a task state transition."""

    COMPLETE = "complete"
    SHOULD_RETRY = "should_retry"
    EXCEEDED_RETRY_LIMIT = "exceeded_retry_limit"


@dataclass
class TaskAttempt:
    """Record of a single task execution attempt."""

    attempt_id: int
    worker_id: WorkerId | None = None
    state: int = cluster_pb2.TASK_STATE_PENDING
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    exit_code: int | None = None
    error: str | None = None
    is_worker_failure: bool = False


@dataclass
class Task:
    """Controller's representation of a task within a job.

    Tasks are created when a job is submitted:
    - Job with replicas=1 creates 1 task
    - Job with replicas=N creates N tasks

    State transitions are handled via transition(), which updates internal
    state and returns a TaskTransitionResult indicating what the caller should do.
    """

    task_id: TaskId
    job_id: JobId
    task_index: int

    state: int = cluster_pb2.TASK_STATE_PENDING
    worker_id: WorkerId | None = None

    # Retry tracking (per-task)
    failure_count: int = 0
    preemption_count: int = 0
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Attempt tracking
    current_attempt_id: int = 0
    attempts: list[TaskAttempt] = field(default_factory=list)

    # Timestamps
    submitted_at_ms: int = 0
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    error: str | None = None
    exit_code: int | None = None

    def mark_dispatched(self, worker_id: WorkerId, now_ms: int) -> None:
        """Mark task as dispatched to a worker. Called BEFORE the RPC."""
        self.state = cluster_pb2.TASK_STATE_RUNNING
        self.worker_id = worker_id
        self.started_at_ms = now_ms

    def revert_dispatch(self) -> None:
        """Revert dispatch if RPC fails."""
        self.state = cluster_pb2.TASK_STATE_PENDING
        self.worker_id = None
        self.started_at_ms = None

    def transition(
        self,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> TaskTransitionResult:
        """Transition to a new state.

        For failure states, handles retry logic internally:
        - Increments appropriate counter (failure_count or preemption_count)
        - If retries remaining: resets to PENDING, returns SHOULD_RETRY
        - If no retries: stays in failure state, returns EXCEEDED_RETRY_LIMIT

        Args:
            new_state: Target state
            now_ms: Current timestamp
            is_worker_failure: True if failure due to worker death (preemption)
            error: Error message for failure states
            exit_code: Exit code for completed tasks

        Returns:
            TaskTransitionResult indicating what caller should do
        """
        if new_state in (
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        ):
            return self._handle_failure(now_ms, is_worker_failure, error, exit_code)

        if new_state == cluster_pb2.TASK_STATE_SUCCEEDED:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.exit_code = exit_code or 0
            return TaskTransitionResult.COMPLETE

        if new_state == cluster_pb2.TASK_STATE_KILLED:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.error = error
            return TaskTransitionResult.COMPLETE

        if new_state == cluster_pb2.TASK_STATE_UNSCHEDULABLE:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.error = error or "Scheduling timeout exceeded"
            return TaskTransitionResult.COMPLETE

        # Non-terminal states (BUILDING, RUNNING)
        self.state = new_state
        return TaskTransitionResult.COMPLETE

    def _handle_failure(
        self,
        now_ms: int,
        is_worker_failure: bool,
        error: str | None,
        exit_code: int | None,
    ) -> TaskTransitionResult:
        """Handle failure with retry logic. Either resets for retry or marks as terminal failure."""
        if is_worker_failure:
            self.preemption_count += 1
            can_retry = self.preemption_count <= self.max_retries_preemption
        else:
            self.failure_count += 1
            can_retry = self.failure_count <= self.max_retries_failure

        if can_retry:
            self._reset_for_retry(is_worker_failure=is_worker_failure)
            return TaskTransitionResult.SHOULD_RETRY
        else:
            self.state = cluster_pb2.TASK_STATE_WORKER_FAILED if is_worker_failure else cluster_pb2.TASK_STATE_FAILED
            self.finished_at_ms = now_ms
            self.error = error
            self.exit_code = exit_code
            return TaskTransitionResult.EXCEEDED_RETRY_LIMIT

    def _reset_for_retry(self, *, is_worker_failure: bool) -> None:
        """Reset state for retry attempt, preserving history."""
        if self.started_at_ms is not None:
            self.attempts.append(
                TaskAttempt(
                    attempt_id=self.current_attempt_id,
                    worker_id=self.worker_id,
                    state=self.state,
                    started_at_ms=self.started_at_ms,
                    finished_at_ms=self.finished_at_ms,
                    exit_code=self.exit_code,
                    error=self.error,
                    is_worker_failure=is_worker_failure,
                )
            )

        self.current_attempt_id += 1
        self.state = cluster_pb2.TASK_STATE_PENDING
        self.worker_id = None
        self.started_at_ms = None
        self.finished_at_ms = None
        self.error = None
        self.exit_code = None

    def is_finished(self) -> bool:
        """Check if task is in a terminal state."""
        return self.state in (
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
            cluster_pb2.TASK_STATE_UNSCHEDULABLE,
        )

    @property
    def total_attempts(self) -> int:
        """Total number of attempts including current."""
        return len(self.attempts) + 1
