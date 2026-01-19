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

State ownership follows a clear hierarchy:
- TaskAttempt owns execution state (timestamps, exit codes, errors)
- Task state is derived from the current attempt
- Task tracks retry policy and counts at the task level
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

    Tasks track identity and retry policy. Execution state (worker assignments,
    timestamps, results) is stored in TaskAttempt objects and accessed via properties.

    State transitions are handled via handle_attempt_result(), which updates the
    current attempt's state and returns a TaskTransitionResult indicating what
    the caller should do.
    """

    task_id: TaskId
    job_id: JobId
    task_index: int

    # Retry policy (immutable after creation)
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Retry counters (task-level, not attempt-level)
    failure_count: int = 0
    preemption_count: int = 0

    # Attempt tracking
    attempts: list[TaskAttempt] = field(default_factory=list)

    # Submission timestamp (distinct from attempt start times)
    submitted_at_ms: int = 0

    # --- Properties that delegate to current attempt ---

    @property
    def current_attempt(self) -> TaskAttempt | None:
        """The most recent attempt, or None if no attempts yet."""
        return self.attempts[-1] if self.attempts else None

    @property
    def current_attempt_id(self) -> int:
        """ID of current attempt (0-indexed), or -1 if no attempts."""
        return len(self.attempts) - 1 if self.attempts else -1

    @property
    def state(self) -> int:
        """Task state is the state of the current attempt, or PENDING if none."""
        if not self.attempts:
            return cluster_pb2.TASK_STATE_PENDING
        return self.attempts[-1].state

    @state.setter
    def state(self, value: int) -> None:
        """Set task state - for backward compatibility, will be removed in Stage 4.

        Creates a synthetic attempt if none exists (e.g., for killing PENDING tasks).
        """
        if not self.attempts:
            # Create a synthetic attempt for tasks that were never dispatched
            self.attempts.append(TaskAttempt(attempt_id=0, state=value))
        else:
            self.attempts[-1].state = value

    @property
    def worker_id(self) -> WorkerId | None:
        """Worker from current attempt, if any."""
        if not self.attempts:
            return None
        return self.attempts[-1].worker_id

    @property
    def started_at_ms(self) -> int | None:
        """Start time of current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].started_at_ms

    @started_at_ms.setter
    def started_at_ms(self, value: int | None) -> None:
        """Set start time - for backward compatibility, will be removed in Stage 4."""
        if not self.attempts:
            self.attempts.append(TaskAttempt(attempt_id=0, started_at_ms=value))
        else:
            self.attempts[-1].started_at_ms = value

    @property
    def finished_at_ms(self) -> int | None:
        """Finish time of current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].finished_at_ms

    @finished_at_ms.setter
    def finished_at_ms(self, value: int | None) -> None:
        """Set finish time - for backward compatibility, will be removed in Stage 4."""
        if not self.attempts:
            self.attempts.append(TaskAttempt(attempt_id=0, finished_at_ms=value))
        else:
            self.attempts[-1].finished_at_ms = value

    @property
    def exit_code(self) -> int | None:
        """Exit code from current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].exit_code

    @exit_code.setter
    def exit_code(self, value: int | None) -> None:
        """Set exit code - for backward compatibility, will be removed in Stage 4."""
        if not self.attempts:
            self.attempts.append(TaskAttempt(attempt_id=0, exit_code=value))
        else:
            self.attempts[-1].exit_code = value

    @property
    def error(self) -> str | None:
        """Error from current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].error

    @error.setter
    def error(self, value: str | None) -> None:
        """Set error - for backward compatibility, will be removed in Stage 4."""
        if not self.attempts:
            self.attempts.append(TaskAttempt(attempt_id=0, error=value))
        else:
            self.attempts[-1].error = value

    # --- Attempt management methods ---

    def create_attempt(self, worker_id: WorkerId, now_ms: int) -> TaskAttempt:
        """Create a new attempt for this task.

        Called when the scheduler assigns this task to a worker.
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
        return attempt

    def revert_attempt(self) -> None:
        """Remove the current attempt if dispatch RPC fails.

        Called when we created an attempt but the RPC to dispatch
        to the worker failed, so we need to undo.
        """
        if self.attempts:
            self.attempts.pop()

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

        Transitions the current attempt's state and handles retry logic:
        - If retriable failure: returns SHOULD_RETRY (task stays in list, ready for new attempt)
        - If terminal success: returns COMPLETE
        - If exhausted retries: returns EXCEEDED_RETRY_LIMIT

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
            # Transition the attempt to the failure state
            attempt.transition(
                new_state,
                now_ms,
                exit_code=exit_code,
                error=error,
                is_worker_failure=is_worker_failure,
            )
            return self._handle_failure(is_worker_failure)

        # For success, set exit_code to 0 if not provided
        if new_state == cluster_pb2.TASK_STATE_SUCCEEDED:
            attempt.transition(
                new_state,
                now_ms,
                exit_code=exit_code if exit_code is not None else 0,
                error=error,
                is_worker_failure=is_worker_failure,
            )
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
            return TaskTransitionResult.COMPLETE

        # Non-terminal states (BUILDING, RUNNING)
        attempt.transition(
            new_state,
            now_ms,
            exit_code=exit_code,
            error=error,
            is_worker_failure=is_worker_failure,
        )
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

    # --- Backward compatibility aliases ---
    # These will be removed once callers are updated (Stages 4+)

    def mark_dispatched(self, worker_id: WorkerId, now_ms: int) -> None:
        """Alias for create_attempt() - will be removed in Stage 4."""
        self.create_attempt(worker_id, now_ms)

    def revert_dispatch(self) -> None:
        """Alias for revert_attempt() - will be removed in Stage 4."""
        self.revert_attempt()

    def transition(
        self,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> TaskTransitionResult:
        """Backward compatible transition - will be removed in Stage 4.

        Preserves old behavior:
        - Allows transitioning tasks without attempts (e.g., killing PENDING tasks)
        - Resets state to PENDING when SHOULD_RETRY
        The new handle_attempt_result() requires an attempt and keeps it terminal.
        """
        # Handle case where task has no attempts (e.g., killing a PENDING task)
        if not self.attempts:
            self.state = new_state
            if new_state in TERMINAL_TASK_STATES:
                self.finished_at_ms = now_ms
                self.error = error
                self.exit_code = exit_code
            return TaskTransitionResult.COMPLETE

        result = self.handle_attempt_result(
            new_state,
            now_ms,
            is_worker_failure=is_worker_failure,
            error=error,
            exit_code=exit_code,
        )
        # Preserve old behavior: reset state to PENDING on retry
        if result == TaskTransitionResult.SHOULD_RETRY:
            self.state = cluster_pb2.TASK_STATE_PENDING
            self.started_at_ms = None
            self.finished_at_ms = None
            self.error = None
            self.exit_code = None
        return result
