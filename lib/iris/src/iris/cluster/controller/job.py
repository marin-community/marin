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

"""Self-contained Job with state transition logic and retry tracking."""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from iris.cluster.controller.task import TERMINAL_TASK_STATES, Task
from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2


class TransitionResult(Enum):
    """Result of a state transition."""

    COMPLETE = "complete"  # Transition succeeded, no action needed
    SHOULD_RETRY = "should_retry"  # Job should be re-queued for retry
    EXCEEDED_RETRY_LIMIT = "exceeded_retry_limit"  # Exceeded retry limit, stay in terminal state


@dataclass
class JobAttempt:
    """Record of a single job execution attempt."""

    attempt_id: int
    worker_id: WorkerId | None = None
    state: int = cluster_pb2.JOB_STATE_PENDING
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    exit_code: int | None = None
    error: str | None = None
    is_worker_failure: bool = False


@dataclass
class Job:
    """Job with self-contained state transitions.

    State transitions are handled via transition(), which updates internal
    state and returns a TransitionResult indicating what the caller should do.

    Example:
        job = Job(job_id=..., request=...)

        # Dispatch to worker
        job.mark_dispatched(worker_id, now_ms)

        # Worker reports failure
        result = job.transition(JOB_STATE_FAILED, now_ms, is_worker_failure=False)
        if result == TransitionResult.SHOULD_RETRY:
            queue.add(job)  # Caller handles re-queueing
    """

    job_id: JobId
    request: cluster_pb2.Controller.LaunchJobRequest
    state: int = cluster_pb2.JOB_STATE_PENDING
    worker_id: WorkerId | None = None

    # Retry tracking
    failure_count: int = 0
    preemption_count: int = 0
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Gang scheduling
    gang_id: str | None = None

    # Hierarchical job tracking
    parent_job_id: JobId | None = None

    # Attempt tracking
    current_attempt_id: int = 0
    attempts: list[JobAttempt] = field(default_factory=list)

    # Timestamps
    submitted_at_ms: int = 0
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    error: str | None = None
    exit_code: int | None = None

    # Incremental task state tracking
    num_tasks: int = 0
    task_state_counts: Counter[int] = field(default_factory=Counter)

    # --- State Transitions ---

    def mark_dispatched(self, worker_id: WorkerId, now_ms: int) -> None:
        """Mark job as dispatched to a worker. Called BEFORE the RPC."""
        self.state = cluster_pb2.JOB_STATE_RUNNING
        self.worker_id = worker_id
        self.started_at_ms = now_ms

    def revert_dispatch(self) -> None:
        self.state = cluster_pb2.JOB_STATE_PENDING
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
    ) -> TransitionResult:
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
            exit_code: Exit code for completed jobs

        Returns:
            TransitionResult indicating what caller should do
        """
        # Handle failure states with retry logic
        if new_state in (
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_WORKER_FAILED,
        ):
            return self._handle_failure(now_ms, is_worker_failure, error, exit_code)

        # Non-failure terminal states
        if new_state == cluster_pb2.JOB_STATE_SUCCEEDED:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.exit_code = exit_code or 0
            return TransitionResult.COMPLETE

        if new_state == cluster_pb2.JOB_STATE_KILLED:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.error = error
            return TransitionResult.COMPLETE

        if new_state == cluster_pb2.JOB_STATE_UNSCHEDULABLE:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.error = error or f"Scheduling timeout exceeded ({self.request.scheduling_timeout_seconds}s)"
            return TransitionResult.COMPLETE

        # Non-terminal states (BUILDING, RUNNING)
        self.state = new_state
        return TransitionResult.COMPLETE

    def _handle_failure(
        self,
        now_ms: int,
        is_worker_failure: bool,
        error: str | None,
        exit_code: int | None,
    ) -> TransitionResult:
        """Handle failure with retry logic. Either resets for retry or marks as terminal failure."""
        if is_worker_failure:
            self.preemption_count += 1
            can_retry = self.preemption_count <= self.max_retries_preemption
        else:
            self.failure_count += 1
            can_retry = self.failure_count <= self.max_retries_failure

        if can_retry:
            self._reset_for_retry(is_worker_failure=is_worker_failure)
            return TransitionResult.SHOULD_RETRY
        else:
            # Terminal failure
            self.state = cluster_pb2.JOB_STATE_WORKER_FAILED if is_worker_failure else cluster_pb2.JOB_STATE_FAILED
            self.finished_at_ms = now_ms
            self.error = error
            self.exit_code = exit_code
            return TransitionResult.EXCEEDED_RETRY_LIMIT

    def _reset_for_retry(self, *, is_worker_failure: bool) -> None:
        """Reset state for retry attempt, preserving history."""
        # Save current attempt to history if it was actually started
        if self.started_at_ms is not None:
            self.attempts.append(
                JobAttempt(
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

        # Increment attempt counter
        self.current_attempt_id += 1

        # Reset current state
        self.state = cluster_pb2.JOB_STATE_PENDING
        self.worker_id = None
        self.started_at_ms = None
        self.finished_at_ms = None
        self.error = None
        self.exit_code = None

    # --- Task State Tracking ---

    def on_task_transition(self, old_state: int | None, new_state: int, now_ms: int) -> int | None:
        """Update counts for a single task transition.

        Args:
            old_state: Previous task state, or None if new task
            new_state: New task state
            now_ms: Current timestamp in milliseconds

        Returns:
            New job state if changed, None otherwise
        """
        if old_state is not None:
            self.task_state_counts[old_state] -= 1
        self.task_state_counts[new_state] += 1
        return self._compute_job_state(now_ms)

    def _compute_job_state(self, now_ms: int) -> int | None:
        """Derive job state from counts. O(1) - no task iteration.

        Returns:
            New job state if changed, None otherwise
        """
        counts = self.task_state_counts

        # Job succeeds when all tasks succeed
        if counts[cluster_pb2.TASK_STATE_SUCCEEDED] == self.num_tasks:
            return cluster_pb2.JOB_STATE_SUCCEEDED

        # Only actual failures count (not preemptions/worker failures)
        max_task_failures = self.request.max_task_failures
        if counts[cluster_pb2.TASK_STATE_FAILED] > max_task_failures:
            return cluster_pb2.JOB_STATE_FAILED

        # Job unschedulable if any task is unschedulable
        if counts[cluster_pb2.TASK_STATE_UNSCHEDULABLE] > 0:
            return cluster_pb2.JOB_STATE_UNSCHEDULABLE

        # Job killed if any task was killed (and we're not already in a terminal state)
        if counts[cluster_pb2.TASK_STATE_KILLED] > 0 and not self.is_finished():
            return cluster_pb2.JOB_STATE_KILLED

        # Job is RUNNING if any task is running
        if counts[cluster_pb2.TASK_STATE_RUNNING] > 0 and self.state != cluster_pb2.JOB_STATE_RUNNING:
            if self.started_at_ms is None:
                self.started_at_ms = now_ms
            return cluster_pb2.JOB_STATE_RUNNING

        return None

    @property
    def finished_task_count(self) -> int:
        """Count of tasks in terminal states."""
        return sum(self.task_state_counts[s] for s in TERMINAL_TASK_STATES)

    # --- State Queries ---

    def is_finished(self) -> bool:
        return self.state in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
            cluster_pb2.JOB_STATE_WORKER_FAILED,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        )

    def can_retry_failure(self) -> bool:
        return self.failure_count < self.max_retries_failure

    def can_retry_preemption(self) -> bool:
        return self.preemption_count < self.max_retries_preemption

    @property
    def total_attempts(self) -> int:
        return len(self.attempts) + 1


def handle_gang_failure(
    jobs: list[Job],
    now_ms: int,
    is_worker_failure: bool,
    error: str,
) -> list[Job]:
    """Handle gang failure with all-or-nothing retry.

    All jobs in gang must have retries remaining for any to be retried.
    This function coordinates multiple jobs, so it lives outside the Job class.

    Args:
        jobs: All jobs in the gang
        now_ms: Current timestamp
        is_worker_failure: Type of failure
        error: Error message

    Returns:
        List of jobs to re-queue (empty if no retry)
    """
    if not jobs:
        return []

    # Check ALL jobs can retry (all-or-nothing)
    if is_worker_failure:
        can_retry = all(j.can_retry_preemption() for j in jobs)
    else:
        can_retry = all(j.can_retry_failure() for j in jobs)

    if not can_retry:
        # Mark all running jobs as killed
        for job in jobs:
            if job.state == cluster_pb2.JOB_STATE_RUNNING:
                job.transition(cluster_pb2.JOB_STATE_KILLED, now_ms, error=error)
        return []

    # Retry all jobs
    to_requeue = []
    for job in jobs:
        result = job.transition(
            cluster_pb2.JOB_STATE_FAILED,
            now_ms,
            is_worker_failure=is_worker_failure,
            error=error,
        )
        if result == TransitionResult.SHOULD_RETRY:
            to_requeue.append(job)

    return to_requeue


def expand_job_to_tasks(job: Job, now_ms: int) -> list[Task]:
    """Expand a job into its constituent tasks based on replicas.

    Jobs with replicas=N expand into N tasks. Each task has a unique ID
    of the form "{job_id}/task-{index}" where index is 0-based.

    Args:
        job: The job to expand
        now_ms: Current timestamp in milliseconds

    Returns:
        List of Task objects for this job
    """
    num_replicas = job.request.resources.replicas or 1
    tasks = []

    for i in range(num_replicas):
        task_id = TaskId(f"{job.job_id}/task-{i}")
        task = Task(
            task_id=task_id,
            job_id=job.job_id,
            task_index=i,
            max_retries_failure=job.max_retries_failure,
            max_retries_preemption=job.max_retries_preemption,
            submitted_at_ms=now_ms,
        )
        tasks.append(task)

    return tasks
