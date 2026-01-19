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

Task owns its state directly. Attempts are pure execution history:
- PENDING: no attempts, or all attempts are terminal (awaiting retry)
- RUNNING: last attempt is non-terminal
- SUCCEEDED/FAILED/KILLED: terminal state, possibly with empty attempts (if killed before dispatch)
"""

from dataclasses import dataclass, field
from enum import Enum

from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2

TERMINAL_TASK_STATES: frozenset[int] = frozenset(
    {
        cluster_pb2.TASK_STATE_SUCCEEDED,
        cluster_pb2.TASK_STATE_FAILED,
        cluster_pb2.TASK_STATE_KILLED,
        cluster_pb2.TASK_STATE_UNSCHEDULABLE,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
    }
)


class TaskTransitionResult(Enum):
    """Result of a task state transition."""

    COMPLETE = "complete"
    SHOULD_RETRY = "should_retry"
    EXCEEDED_RETRY_LIMIT = "exceeded_retry_limit"


@dataclass
class TaskAttempt:
    """Record of a single task execution attempt.

    An attempt represents one try at executing a task on a specific worker.
    All execution-related state (timestamps, exit codes, errors) lives here.
    """

    attempt_id: int
    worker_id: WorkerId | None = None
    state: int = cluster_pb2.TASK_STATE_PENDING

    # Timing
    created_at_ms: int = 0
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    # Result
    exit_code: int | None = None
    error: str | None = None
    is_worker_failure: bool = False

    def transition(
        self,
        new_state: int,
        now_ms: int,
        *,
        exit_code: int | None = None,
        error: str | None = None,
        is_worker_failure: bool = False,
    ) -> None:
        """Transition this attempt to a new state."""
        self.state = new_state

        if new_state == cluster_pb2.TASK_STATE_RUNNING:
            self.started_at_ms = now_ms

        if new_state in TERMINAL_TASK_STATES:
            self.finished_at_ms = now_ms
            self.exit_code = exit_code
            self.error = error
            self.is_worker_failure = is_worker_failure

    def is_terminal(self) -> bool:
        """Check if this attempt is in a terminal state."""
        return self.state in TERMINAL_TASK_STATES


@dataclass
class Task:
    """Controller's representation of a task within a job.

    Task owns its state directly - these are dataclass fields, not delegated to attempts.
    Attempts are pure execution history, tracking each time a task was dispatched to a worker.

    State transitions are handled via handle_attempt_result(), which updates both
    the attempt and task-level state, returning a TaskTransitionResult indicating
    what the caller should do.
    """

    task_id: TaskId
    job_id: JobId
    task_index: int

    # Task owns its state directly
    state: int = cluster_pb2.TASK_STATE_PENDING
    error: str | None = None
    exit_code: int | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    # Retry policy (immutable after creation)
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Retry counters (task-level, not attempt-level)
    failure_count: int = 0
    preemption_count: int = 0

    # Attempt tracking - pure execution history
    attempts: list[TaskAttempt] = field(default_factory=list)

    # Submission timestamp (distinct from attempt start times)
    submitted_at_ms: int = 0

    # --- Read-only properties that derive from attempts ---

    @property
    def current_attempt(self) -> TaskAttempt | None:
        """The most recent attempt, or None if no attempts yet."""
        return self.attempts[-1] if self.attempts else None

    @property
    def current_attempt_id(self) -> int:
        """ID of current attempt (0-indexed), or -1 if no attempts."""
        return len(self.attempts) - 1 if self.attempts else -1

    @property
    def worker_id(self) -> WorkerId | None:
        """Worker from the most recent attempt, if any.

        Returns the worker ID from the last attempt regardless of whether
        the attempt is terminal. This is used for reporting which worker
        ran (or is running) the task.
        """
        if self.attempts:
            return self.attempts[-1].worker_id
        return None

    # --- Attempt management methods ---

    def create_attempt(self, worker_id: WorkerId, now_ms: int) -> TaskAttempt:
        """Create a new attempt for this task.

        Called when the scheduler assigns this task to a worker.
        Updates task state to RUNNING and sets started_at_ms on first attempt.
        Returns the new attempt so caller can track it.
        """
        attempt = TaskAttempt(
            attempt_id=len(self.attempts),
            worker_id=worker_id,
            state=cluster_pb2.TASK_STATE_RUNNING,
            created_at_ms=now_ms,
            started_at_ms=now_ms,
        )
        self.attempts.append(attempt)

        # Update task-level state
        self.state = cluster_pb2.TASK_STATE_RUNNING
        if self.started_at_ms is None:
            self.started_at_ms = now_ms

        return attempt

    def revert_attempt(self) -> None:
        """Remove the current attempt if dispatch RPC fails.

        Called when we created an attempt but the RPC to dispatch
        to the worker failed, so we need to undo. Resets state to PENDING
        since create_attempt() set it to RUNNING.
        """
        if self.attempts:
            self.attempts.pop()
            self.state = cluster_pb2.TASK_STATE_PENDING

    def handle_attempt_result(
        self,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> TaskTransitionResult:
        """Handle a state report for the current attempt.

        Updates both the attempt and task-level state. Handles retry logic:
        - If retriable failure: returns SHOULD_RETRY, task state reflects failure but is schedulable
        - If terminal success: returns COMPLETE, task state is terminal
        - If exhausted retries: returns EXCEEDED_RETRY_LIMIT, task state is terminal

        Does NOT create new attempts - that's the scheduler's job.

        Args:
            new_state: Target state
            now_ms: Current timestamp
            is_worker_failure: True if failure due to worker death (preemption)
            error: Error message for failure states
            exit_code: Exit code for completed tasks

        Returns:
            TaskTransitionResult indicating what caller should do
        """
        if not self.attempts:
            raise ValueError("Cannot handle result without an attempt")

        attempt = self.attempts[-1]

        # Handle failure states specially for retry logic
        if new_state in (
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        ):
            attempt.transition(
                new_state,
                now_ms,
                exit_code=exit_code,
                error=error,
                is_worker_failure=is_worker_failure,
            )
            result = self._handle_failure(is_worker_failure)

            # Update task-level state to reflect the failure
            self.state = new_state
            if result == TaskTransitionResult.EXCEEDED_RETRY_LIMIT:
                # Terminal: record final outcome
                self.error = error
                self.exit_code = exit_code
                self.finished_at_ms = now_ms
            # For SHOULD_RETRY, we don't set finished_at_ms since task isn't truly finished
            return result

        # For success, set exit_code to 0 if not provided
        if new_state == cluster_pb2.TASK_STATE_SUCCEEDED:
            final_exit_code = exit_code if exit_code is not None else 0
            attempt.transition(
                new_state,
                now_ms,
                exit_code=final_exit_code,
                error=error,
                is_worker_failure=is_worker_failure,
            )
            # Update task-level state
            self.state = new_state
            self.exit_code = final_exit_code
            self.error = error
            self.finished_at_ms = now_ms
            return TaskTransitionResult.COMPLETE

        # For other terminal states (KILLED, UNSCHEDULABLE)
        if new_state in (cluster_pb2.TASK_STATE_KILLED, cluster_pb2.TASK_STATE_UNSCHEDULABLE):
            actual_error = error
            if new_state == cluster_pb2.TASK_STATE_UNSCHEDULABLE and error is None:
                actual_error = "Scheduling timeout exceeded"
            attempt.transition(
                new_state,
                now_ms,
                exit_code=exit_code,
                error=actual_error,
                is_worker_failure=is_worker_failure,
            )
            # Update task-level state
            self.state = new_state
            self.error = actual_error
            self.exit_code = exit_code
            self.finished_at_ms = now_ms
            return TaskTransitionResult.COMPLETE

        # Non-terminal states (BUILDING, RUNNING)
        attempt.transition(
            new_state,
            now_ms,
            exit_code=exit_code,
            error=error,
            is_worker_failure=is_worker_failure,
        )
        # Update task-level state for non-terminal transitions
        self.state = new_state
        return TaskTransitionResult.COMPLETE

    def _handle_failure(self, is_worker_failure: bool) -> TaskTransitionResult:
        """Determine if task should retry after a failure.

        Does NOT reset task state - current attempt stays terminal.
        Scheduler will create new attempt when it reassigns the task.
        """
        if is_worker_failure:
            self.preemption_count += 1
            can_retry = self.preemption_count <= self.max_retries_preemption
        else:
            self.failure_count += 1
            can_retry = self.failure_count <= self.max_retries_failure

        if can_retry:
            return TaskTransitionResult.SHOULD_RETRY
        else:
            return TaskTransitionResult.EXCEEDED_RETRY_LIMIT

    def is_finished(self) -> bool:
        """Check if task is truly finished (terminal with no retries remaining).

        Returns True if:
        - Task succeeded, was killed, or is unschedulable (non-retriable states)
        - Task failed and has exhausted failure retries
        - Task worker-failed and has exhausted preemption retries
        """
        state = self.state
        if state == cluster_pb2.TASK_STATE_SUCCEEDED:
            return True
        if state in (cluster_pb2.TASK_STATE_KILLED, cluster_pb2.TASK_STATE_UNSCHEDULABLE):
            return True
        if state == cluster_pb2.TASK_STATE_FAILED:
            return self.failure_count > self.max_retries_failure
        if state == cluster_pb2.TASK_STATE_WORKER_FAILED:
            return self.preemption_count > self.max_retries_preemption
        return False

    def can_be_scheduled(self) -> bool:
        """Check if task is ready to be scheduled.

        A task can be scheduled if:
        - It has no attempts yet (fresh task), or
        - Its current attempt is terminal AND it should retry
        """
        if not self.attempts:
            return True
        return self.attempts[-1].is_terminal() and not self.is_finished()

    @property
    def total_attempts(self) -> int:
        """Total number of attempts."""
        return len(self.attempts)
