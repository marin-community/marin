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

"""Controller core data structures and thread-safe state container.

This module contains all controller state objects:
- Task: unit of execution that runs on a single worker
- TaskAttempt: record of a single task execution attempt
- Job: collection of tasks with shared configuration
- ControllerWorker: controller's view of a worker
- ControllerEndpoint: service discovery endpoint
- ControllerState: thread-safe state container

State transitions are handled via methods on each class, which update internal
state and return results indicating what the caller should do.
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock

from iris.cluster.controller.events import Event, EventType, TransactionLog
from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)

# =============================================================================
# Device Helper Functions
# =============================================================================


def get_device_type(device: cluster_pb2.DeviceConfig) -> str:
    """Extract device type from config."""
    if device.HasField("cpu"):
        return "cpu"
    elif device.HasField("gpu"):
        return "gpu"
    elif device.HasField("tpu"):
        return "tpu"
    return "cpu"


def get_device_variant(device: cluster_pb2.DeviceConfig) -> str | None:
    """Extract device variant (e.g., GPU model) from config."""
    if device.HasField("gpu"):
        return device.gpu.variant if device.gpu.variant else None
    elif device.HasField("tpu"):
        return device.tpu.variant if device.tpu.variant else None
    return None


def get_gpu_count(device: cluster_pb2.DeviceConfig) -> int:
    """Extract GPU count from config."""
    if device.HasField("gpu"):
        return device.gpu.count or 1
    return 0


# =============================================================================
# Task State Definitions
# =============================================================================

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


class JobTransitionResult(Enum):
    """Result of a state transition."""

    COMPLETE = "complete"  # Transition succeeded, no action needed
    SHOULD_RETRY = "should_retry"  # Job should be re-queued for retry
    EXCEEDED_RETRY_LIMIT = "exceeded_retry_limit"  # Exceeded retry limit, stay in terminal state


# =============================================================================
# Task Classes
# =============================================================================


@dataclass
class ControllerTaskAttempt:
    """Record of a single task execution attempt.

    An attempt represents one try at executing a task on a specific worker.
    All execution-related state (timestamps, exit codes, errors) lives here.

    Whether this was a worker failure is encoded in the state itself:
    TASK_STATE_WORKER_FAILED indicates worker died, TASK_STATE_FAILED indicates task error.
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

    def transition(
        self,
        new_state: int,
        *,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> None:
        """Transition this attempt to a new state."""
        self.state = new_state
        ts = now_ms()

        if new_state == cluster_pb2.TASK_STATE_RUNNING:
            self.started_at_ms = ts

        if new_state in TERMINAL_TASK_STATES:
            self.finished_at_ms = ts
            self.exit_code = exit_code
            self.error = error

    def is_terminal(self) -> bool:
        """Check if this attempt is in a terminal state."""
        return self.state in TERMINAL_TASK_STATES

    @property
    def is_worker_failure(self) -> bool:
        """Whether this attempt failed due to worker death (derived from state)."""
        return self.state == cluster_pb2.TASK_STATE_WORKER_FAILED


@dataclass
class ControllerTask:
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
    attempts: list[ControllerTaskAttempt] = field(default_factory=list)

    # Submission timestamp (distinct from attempt start times)
    submitted_at_ms: int = 0

    # --- Read-only properties that derive from attempts ---

    @property
    def current_attempt(self) -> ControllerTaskAttempt | None:
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

    def create_attempt(self, worker_id: WorkerId) -> ControllerTaskAttempt:
        """Create a new attempt for this task.

        Called when the scheduler assigns this task to a worker.
        Updates task state to RUNNING and sets started_at_ms on first attempt.
        Returns the new attempt so caller can track it.
        """
        ts = now_ms()
        attempt = ControllerTaskAttempt(
            attempt_id=len(self.attempts),
            worker_id=worker_id,
            state=cluster_pb2.TASK_STATE_RUNNING,
            created_at_ms=ts,
            started_at_ms=ts,
        )
        self.attempts.append(attempt)

        # Update task-level state
        self.state = cluster_pb2.TASK_STATE_RUNNING
        if self.started_at_ms is None:
            self.started_at_ms = ts

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
        *,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> TaskTransitionResult:
        """Handle a state report for the current attempt.

        Updates both the attempt and task-level state. Handles retry logic:
        - If retriable failure: returns SHOULD_RETRY, task state reflects failure but is schedulable
        - If terminal success: returns COMPLETE, task state is terminal
        - If exhausted retries: returns EXCEEDED_RETRY_LIMIT, task state is terminal

        Does NOT create new attempts - that's the scheduler's job.

        Whether this is a worker failure is derived from the target state:
        - TASK_STATE_WORKER_FAILED -> uses preemption retry budget
        - TASK_STATE_FAILED -> uses failure retry budget

        Args:
            new_state: Target state
            error: Error message for failure states
            exit_code: Exit code for completed tasks

        Returns:
            TaskTransitionResult indicating what caller should do
        """
        if not self.attempts:
            self.state = new_state
            if new_state in TERMINAL_TASK_STATES:
                self.finished_at_ms = now_ms()
                self.error = error
                self.exit_code = exit_code
            return TaskTransitionResult.COMPLETE

        attempt = self.attempts[-1]

        # Handle failure states specially for retry logic
        if new_state in (
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        ):
            attempt.transition(
                new_state,
                exit_code=exit_code,
                error=error,
            )
            result = self._handle_failure(new_state)

            # Update task-level state to reflect the failure
            self.state = new_state
            if result == TaskTransitionResult.EXCEEDED_RETRY_LIMIT:
                # Terminal: record final outcome
                self.error = error
                self.exit_code = exit_code
                self.finished_at_ms = now_ms()
            return result

        # For success, set exit_code to 0 if not provided
        if new_state == cluster_pb2.TASK_STATE_SUCCEEDED:
            final_exit_code = exit_code if exit_code is not None else 0
            attempt.transition(
                new_state,
                exit_code=final_exit_code,
                error=error,
            )
            # Update task-level state
            self.state = new_state
            self.exit_code = final_exit_code
            self.error = error
            self.finished_at_ms = now_ms()
            return TaskTransitionResult.COMPLETE

        # For other terminal states (KILLED, UNSCHEDULABLE)
        if new_state in (cluster_pb2.TASK_STATE_KILLED, cluster_pb2.TASK_STATE_UNSCHEDULABLE):
            actual_error = error
            if new_state == cluster_pb2.TASK_STATE_UNSCHEDULABLE and error is None:
                actual_error = "Scheduling timeout exceeded"
            attempt.transition(
                new_state,
                exit_code=exit_code,
                error=actual_error,
            )
            # Update task-level state
            self.state = new_state
            self.error = actual_error
            self.exit_code = exit_code
            self.finished_at_ms = now_ms()
            return TaskTransitionResult.COMPLETE

        # Non-terminal states (BUILDING, RUNNING)
        attempt.transition(
            new_state,
            exit_code=exit_code,
            error=error,
        )
        # Update task-level state for non-terminal transitions
        self.state = new_state
        return TaskTransitionResult.COMPLETE

    def _handle_failure(self, new_state: int) -> TaskTransitionResult:
        """Determine if task should retry after a failure.

        Does NOT reset task state - current attempt stays terminal.
        Scheduler will create new attempt when it reassigns the task.

        Args:
            new_state: The failure state (TASK_STATE_FAILED or TASK_STATE_WORKER_FAILED)
        """
        if new_state == cluster_pb2.TASK_STATE_WORKER_FAILED:
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


# =============================================================================
# Job Class
# =============================================================================


@dataclass
class ControllerJob:
    """Job with self-contained state transitions.

    State transitions are handled via transition(), which updates internal
    state and returns a JobTransitionResult indicating what the caller should do.

    Example:
        job = ControllerJob(job_id=..., request=...)

        # Job starts running when first task starts
        job.mark_dispatched()

        # Task reports failure
        result = job.transition(JOB_STATE_FAILED, is_worker_failure=False)
        if result == JobTransitionResult.SHOULD_RETRY:
            queue.add(job)  # Caller handles re-queueing
    """

    job_id: JobId
    request: cluster_pb2.Controller.LaunchJobRequest
    state: int = cluster_pb2.JOB_STATE_PENDING

    # Retry tracking
    failure_count: int = 0
    preemption_count: int = 0
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Gang scheduling
    gang_id: str | None = None

    # Hierarchical job tracking
    parent_job_id: JobId | None = None

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

    def mark_dispatched(self) -> None:
        """Mark job as running. Called when first task starts."""
        self.state = cluster_pb2.JOB_STATE_RUNNING
        self.started_at_ms = now_ms()

    def revert_dispatch(self) -> None:
        """Revert dispatch if no tasks actually started."""
        self.state = cluster_pb2.JOB_STATE_PENDING
        self.started_at_ms = None

    def transition(
        self,
        new_state: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> JobTransitionResult:
        """Transition to a new state.

        For failure states, handles retry logic internally:
        - Increments appropriate counter (failure_count or preemption_count)
        - If retries remaining: resets to PENDING, returns SHOULD_RETRY
        - If no retries: stays in failure state, returns EXCEEDED_RETRY_LIMIT

        Args:
            new_state: Target state
            is_worker_failure: True if failure due to worker death (preemption)
            error: Error message for failure states
            exit_code: Exit code for completed jobs

        Returns:
            JobTransitionResult indicating what caller should do
        """
        # Handle failure states with retry logic
        if new_state == cluster_pb2.JOB_STATE_FAILED:
            return self._handle_failure(is_worker_failure, error, exit_code)

        ts = now_ms()

        # Non-failure terminal states
        if new_state == cluster_pb2.JOB_STATE_SUCCEEDED:
            self.state = new_state
            self.finished_at_ms = ts
            self.exit_code = exit_code or 0
            return JobTransitionResult.COMPLETE

        if new_state == cluster_pb2.JOB_STATE_KILLED:
            self.state = new_state
            self.finished_at_ms = ts
            self.error = error
            return JobTransitionResult.COMPLETE

        if new_state == cluster_pb2.JOB_STATE_UNSCHEDULABLE:
            self.state = new_state
            self.finished_at_ms = ts
            self.error = error or f"Scheduling timeout exceeded ({self.request.scheduling_timeout_seconds}s)"
            return JobTransitionResult.COMPLETE

        # Non-terminal states (BUILDING, RUNNING)
        self.state = new_state
        return JobTransitionResult.COMPLETE

    def _handle_failure(
        self,
        is_worker_failure: bool,
        error: str | None,
        exit_code: int | None,
    ) -> JobTransitionResult:
        """Handle failure with retry logic. Either resets for retry or marks as terminal failure."""
        if is_worker_failure:
            self.preemption_count += 1
            can_retry = self.preemption_count <= self.max_retries_preemption
        else:
            self.failure_count += 1
            can_retry = self.failure_count <= self.max_retries_failure

        if can_retry:
            # Reset state for retry
            self.state = cluster_pb2.JOB_STATE_PENDING
            self.started_at_ms = None
            self.finished_at_ms = None
            self.error = None
            self.exit_code = None
            return JobTransitionResult.SHOULD_RETRY
        else:
            # Terminal failure
            self.state = cluster_pb2.JOB_STATE_FAILED
            self.finished_at_ms = now_ms()
            self.error = error
            self.exit_code = exit_code
            return JobTransitionResult.EXCEEDED_RETRY_LIMIT

    # --- Task State Tracking ---

    def on_task_transition(self, old_state: int | None, new_state: int) -> int | None:
        """Update counts for a single task transition.

        Args:
            old_state: Previous task state, or None if new task
            new_state: New task state

        Returns:
            New job state if changed, None otherwise
        """
        if old_state is not None:
            self.task_state_counts[old_state] -= 1
        self.task_state_counts[new_state] += 1
        return self._compute_job_state()

    def _compute_job_state(self) -> int | None:
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
                self.started_at_ms = now_ms()
            return cluster_pb2.JOB_STATE_RUNNING

        # No state change detected
        return None

    @property
    def finished_task_count(self) -> int:
        """Count of tasks in terminal states."""
        return sum(self.task_state_counts[s] for s in TERMINAL_TASK_STATES)

    def is_finished(self) -> bool:
        return self.state in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        )

    def can_retry_failure(self) -> bool:
        return self.failure_count < self.max_retries_failure

    def can_retry_preemption(self) -> bool:
        return self.preemption_count < self.max_retries_preemption

    @property
    def total_attempts(self) -> int:
        """Total number of retries (failure + preemption retries)."""
        return self.failure_count + self.preemption_count


# =============================================================================
# Job Helper Functions
# =============================================================================


def handle_gang_failure(
    jobs: list[ControllerJob],
    is_worker_failure: bool,
    error: str,
) -> list[ControllerJob]:
    """Handle gang failure with all-or-nothing retry.

    All jobs in gang must have retries remaining for any to be retried.
    This function coordinates multiple jobs, so it lives outside the ControllerJob class.

    Args:
        jobs: All jobs in the gang
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
                job.transition(cluster_pb2.JOB_STATE_KILLED, error=error)
        return []

    # Retry all jobs
    to_requeue = []
    for job in jobs:
        result = job.transition(
            cluster_pb2.JOB_STATE_FAILED,
            is_worker_failure=is_worker_failure,
            error=error,
        )
        if result == JobTransitionResult.SHOULD_RETRY:
            to_requeue.append(job)

    return to_requeue


def expand_job_to_tasks(job: ControllerJob) -> list[ControllerTask]:
    """Expand a job into its constituent tasks based on replicas.

    Jobs with replicas=N expand into N tasks. Each task has a unique ID
    of the form "{job_id}/task-{index}" where index is 0-based.

    Args:
        job: The job to expand

    Returns:
        List of ControllerTask objects for this job
    """
    num_replicas = job.request.resources.replicas or 1
    tasks = []

    for i in range(num_replicas):
        task_id = TaskId(f"{job.job_id}/task-{i}")
        task = ControllerTask(
            task_id=task_id,
            job_id=job.job_id,
            task_index=i,
            max_retries_failure=job.max_retries_failure,
            max_retries_preemption=job.max_retries_preemption,
            submitted_at_ms=job.submitted_at_ms,
        )
        tasks.append(task)

    return tasks


# =============================================================================
# Worker and Endpoint Classes
# =============================================================================


@dataclass
class WorkerConfig:
    """Static worker configuration for v0.

    Args:
        worker_id: Unique worker identifier
        address: Worker RPC address (host:port)
        metadata: Worker environment metadata
    """

    worker_id: str
    address: str
    metadata: cluster_pb2.WorkerMetadata


@dataclass
class ControllerWorker:
    """Controller's view of a worker.

    Args:
        worker_id: Unique worker identifier
        address: Worker RPC address (host:port)
        metadata: Worker environment metadata (includes cpu, memory, device, disk)
        healthy: Whether worker is currently healthy
        consecutive_failures: Number of consecutive heartbeat failures
        last_heartbeat_ms: Timestamp of last successful heartbeat
        running_tasks: Set of task IDs currently running on this worker
        committed_cpu: Total CPU cores committed to running tasks
        committed_mem: Total memory bytes committed to running tasks
        committed_gpu: Total GPUs committed to running tasks
    """

    worker_id: WorkerId
    address: str
    metadata: cluster_pb2.WorkerMetadata

    # Health tracking
    healthy: bool = True
    consecutive_failures: int = 0
    last_heartbeat_ms: int = 0

    # Currently running tasks
    running_tasks: set[TaskId] = field(default_factory=set)

    # Committed resources (tracked incrementally)
    committed_cpu: int = 0
    committed_mem: int = 0
    committed_gpu: int = 0

    def get_committed_resources(self) -> tuple[int, int, int]:
        """Return committed (cpu, memory_bytes, gpu_count) for this worker."""
        return (self.committed_cpu, self.committed_mem, self.committed_gpu)

    def assign_task(self, task_id: TaskId, resources: cluster_pb2.ResourceSpecProto) -> None:
        """Assign a task to this worker, updating committed resources."""
        self.running_tasks.add(task_id)
        self.committed_cpu += resources.cpu
        self.committed_mem += resources.memory_bytes
        self.committed_gpu += get_gpu_count(resources.device)

    def unassign_task(self, task_id: TaskId, resources: cluster_pb2.ResourceSpecProto) -> None:
        """Unassign a task from this worker, updating committed resources."""
        self.running_tasks.discard(task_id)
        self.committed_cpu -= resources.cpu
        self.committed_mem -= resources.memory_bytes
        self.committed_gpu -= get_gpu_count(resources.device)


@dataclass
class ControllerEndpoint:
    """An endpoint registered with the controller for service discovery.

    Names are prefixed with the root job ID for namespace isolation: e.g., "abc123/my-actor".

    Args:
        endpoint_id: Unique endpoint identifier
        name: Full prefixed service name (e.g., "abc123/my-actor")
        address: Network address (host:port)
        job_id: Job that registered this endpoint
        metadata: Additional key-value metadata
        registered_at_ms: Timestamp when endpoint was registered
    """

    endpoint_id: str
    name: str  # Full prefixed name: "{root_job_id}/{actor_name}"
    address: str
    job_id: JobId
    metadata: dict[str, str] = field(default_factory=dict)
    registered_at_ms: int = 0


# =============================================================================
# Controller State
# =============================================================================


class ControllerState:
    """Thread-safe controller state managing jobs, tasks, workers, endpoints, and queues.

    State transitions can be performed via the event-driven handle_event() API,
    which provides transaction logging for debugging and supports cascading effects.

    TODO: Migrate legacy methods (transition_task, add_job, mark_task_dispatched, etc.)
    to use handle_event() in a follow-up. The legacy methods remain for backward
    compatibility with existing controller code that has not yet been updated.
    """

    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[JobId, ControllerJob] = {}
        self._tasks: dict[TaskId, ControllerTask] = {}
        self._tasks_by_job: dict[JobId, list[TaskId]] = {}
        self._workers: dict[WorkerId, ControllerWorker] = {}
        self._task_queue: deque[TaskId] = deque()  # FIFO queue of task IDs
        self._gangs: dict[str, set[JobId]] = {}  # gang_id -> job_ids
        self._endpoints: dict[str, ControllerEndpoint] = {}  # endpoint_id -> endpoint
        self._endpoints_by_task: dict[TaskId, set[str]] = {}  # task_id -> endpoint_ids
        self._transactions: deque[TransactionLog] = deque(maxlen=1000)  # Event transaction log

    # =========================================================================
    # Event-Driven State Transitions
    # =========================================================================

    def handle_event(self, event: Event) -> TransactionLog:
        """Main entry point for all event-driven state changes.

        Dispatches to the appropriate handler based on event_type and records
        all actions to a transaction log for debugging.

        Args:
            event: The event to process

        Returns:
            TransactionLog containing all actions taken during event handling
        """
        with self._lock:
            txn = TransactionLog(event=event)

            match event.event_type:
                case EventType.WORKER_REGISTERED:
                    self._on_worker_registered(txn, event)
                case EventType.WORKER_HEARTBEAT:
                    self._on_worker_heartbeat(txn, event)
                case EventType.WORKER_FAILED:
                    self._on_worker_failed(txn, event)
                case EventType.JOB_SUBMITTED:
                    self._on_job_submitted(txn, event)
                case EventType.JOB_CANCELLED:
                    self._on_job_cancelled(txn, event)
                case EventType.TASK_ASSIGNED:
                    self._on_task_assigned(txn, event)
                case EventType.TASK_RUNNING:
                    self._on_task_running(txn, event)
                case EventType.TASK_SUCCEEDED:
                    self._on_task_succeeded(txn, event)
                case EventType.TASK_FAILED:
                    self._on_task_failed(txn, event)
                case EventType.TASK_KILLED:
                    self._on_task_killed(txn, event)
                case EventType.TASK_WORKER_FAILED:
                    self._on_task_worker_failed(txn, event)

            self._transactions.append(txn)
            return txn

    # -------------------------------------------------------------------------
    # Worker Event Handlers
    # -------------------------------------------------------------------------

    def _on_worker_registered(self, txn: TransactionLog, event: Event) -> None:
        assert event.worker_id and event.address and event.metadata
        worker = ControllerWorker(
            worker_id=event.worker_id,
            address=event.address,
            metadata=event.metadata,
            last_heartbeat_ms=event.timestamp_ms or now_ms(),
        )
        self._workers[event.worker_id] = worker
        txn.log("worker_registered", event.worker_id, address=event.address)

    def _on_worker_heartbeat(self, txn: TransactionLog, event: Event) -> None:
        assert event.worker_id and event.timestamp_ms is not None
        worker = self._workers[event.worker_id]
        worker.last_heartbeat_ms = event.timestamp_ms
        worker.healthy = True
        worker.consecutive_failures = 0
        txn.log("heartbeat", event.worker_id)

    def _on_worker_failed(self, txn: TransactionLog, event: Event) -> None:
        assert event.worker_id
        worker = self._workers[event.worker_id]
        worker.healthy = False
        txn.log("worker_failed", event.worker_id, error=event.error)

        # Cascade to running tasks - call handler directly, same transaction
        for task_id in list(worker.running_tasks):
            task = self._tasks[task_id]
            assert task.worker_id == event.worker_id
            if task.state != cluster_pb2.TASK_STATE_RUNNING:
                continue

            cascade_event = Event(
                EventType.TASK_WORKER_FAILED,
                task_id=task_id,
                worker_id=event.worker_id,
                error=f"Worker {event.worker_id} failed: {event.error or 'unknown'}",
            )
            self._on_task_worker_failed(txn, cascade_event)

    # -------------------------------------------------------------------------
    # Job Event Handlers
    # -------------------------------------------------------------------------

    def _on_job_submitted(self, txn: TransactionLog, event: Event) -> None:
        assert event.job_id and event.request
        job = ControllerJob(
            job_id=event.job_id,
            request=event.request,
            submitted_at_ms=event.timestamp_ms or now_ms(),
        )
        self._jobs[event.job_id] = job
        self._tasks_by_job[event.job_id] = []

        tasks = expand_job_to_tasks(job)
        job.num_tasks = len(tasks)

        for task in tasks:
            self._tasks[task.task_id] = task
            self._tasks_by_job[event.job_id].append(task.task_id)
            self._task_queue.append(task.task_id)
            job.task_state_counts[task.state] += 1
            txn.log("task_created", task.task_id, job_id=str(event.job_id))

        if job.gang_id:
            self._gangs.setdefault(job.gang_id, set()).add(event.job_id)

        txn.log("job_submitted", event.job_id, num_tasks=job.num_tasks)

    def _on_job_cancelled(self, txn: TransactionLog, event: Event) -> None:
        assert event.job_id and event.reason
        job = self._jobs[event.job_id]

        for task_id in self._tasks_by_job.get(event.job_id, []):
            task = self._tasks[task_id]
            if task.state in TERMINAL_TASK_STATES:
                continue

            cascade_event = Event(EventType.TASK_KILLED, task_id=task_id, reason=event.reason)
            self._on_task_killed(txn, cascade_event)

        job.state = cluster_pb2.JOB_STATE_KILLED
        job.error = event.reason
        job.finished_at_ms = now_ms()
        txn.log("job_cancelled", event.job_id, reason=event.reason)

    # -------------------------------------------------------------------------
    # Task Event Handlers
    # -------------------------------------------------------------------------

    def _on_task_assigned(self, txn: TransactionLog, event: Event) -> None:
        assert event.task_id and event.worker_id
        task = self._tasks[event.task_id]
        worker = self._workers[event.worker_id]
        job = self._jobs[task.job_id]

        old_state = task.state
        task.create_attempt(event.worker_id)

        # Task is assigned but not yet running - worker needs to report TASK_RUNNING
        task.state = cluster_pb2.TASK_STATE_PENDING
        if task.attempts:
            task.attempts[-1].state = cluster_pb2.TASK_STATE_PENDING

        worker.assign_task(event.task_id, job.request.resources)

        # Remove from queue
        self._task_queue = deque(tid for tid in self._task_queue if tid != event.task_id)

        # Update job counters and state
        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            job.state = new_job_state

        txn.log("task_assigned", event.task_id, worker_id=str(event.worker_id))

    def _on_task_running(self, txn: TransactionLog, event: Event) -> None:
        assert event.task_id
        task = self._tasks[event.task_id]
        job = self._jobs[task.job_id]
        old_state = task.state

        task.state = cluster_pb2.TASK_STATE_RUNNING
        if task.attempts:
            task.attempts[-1].state = cluster_pb2.TASK_STATE_RUNNING
            task.attempts[-1].started_at_ms = now_ms()

        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            job.state = new_job_state

        txn.log("task_running", event.task_id)

    def _on_task_succeeded(self, txn: TransactionLog, event: Event) -> None:
        assert event.task_id
        task = self._tasks[event.task_id]
        job = self._jobs[task.job_id]
        old_state = task.state
        ts = now_ms()

        task.state = cluster_pb2.TASK_STATE_SUCCEEDED
        task.exit_code = event.exit_code if event.exit_code is not None else 0
        task.finished_at_ms = ts

        if task.attempts:
            task.attempts[-1].state = cluster_pb2.TASK_STATE_SUCCEEDED
            task.attempts[-1].exit_code = task.exit_code
            task.attempts[-1].finished_at_ms = ts

        self._finalize_task(task, job, txn)

        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            self._finalize_job_state(job, new_job_state)

        txn.log("task_succeeded", event.task_id, exit_code=task.exit_code)

    def _on_task_failed(self, txn: TransactionLog, event: Event) -> None:
        """Task failed due to task error. Uses failure retry budget."""
        assert event.task_id
        task = self._tasks[event.task_id]
        job = self._jobs[task.job_id]
        old_state = task.state

        task.failure_count += 1
        can_retry = task.failure_count <= task.max_retries_failure

        self._apply_task_failure(
            task,
            job,
            txn,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            error=event.error,
            exit_code=event.exit_code,
            can_retry=can_retry,
        )

        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            self._finalize_job_state(job, new_job_state)

        txn.log("task_failed", event.task_id, error=event.error, can_retry=can_retry)

    def _on_task_killed(self, txn: TransactionLog, event: Event) -> None:
        """Task killed by user/scheduler. No retry."""
        assert event.task_id
        task = self._tasks[event.task_id]
        job = self._jobs[task.job_id]
        old_state = task.state

        self._apply_task_failure(
            task,
            job,
            txn,
            new_state=cluster_pb2.TASK_STATE_KILLED,
            error=event.reason,
            can_retry=False,
        )

        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            self._finalize_job_state(job, new_job_state)

        txn.log("task_killed", event.task_id, reason=event.reason)

    def _on_task_worker_failed(self, txn: TransactionLog, event: Event) -> None:
        """Task failed because worker died. Uses preemption retry budget."""
        assert event.task_id
        task = self._tasks[event.task_id]
        job = self._jobs[task.job_id]
        old_state = task.state

        task.preemption_count += 1
        can_retry = task.preemption_count <= task.max_retries_preemption

        self._apply_task_failure(
            task,
            job,
            txn,
            new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
            error=event.error,
            can_retry=can_retry,
        )

        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            self._finalize_job_state(job, new_job_state)

        txn.log(
            "task_worker_failed",
            event.task_id,
            worker_id=str(event.worker_id) if event.worker_id else None,
            can_retry=can_retry,
        )

    # -------------------------------------------------------------------------
    # Shared Helpers for Event Handlers
    # -------------------------------------------------------------------------

    def _apply_task_failure(
        self,
        task: ControllerTask,
        job: ControllerJob,
        txn: TransactionLog,
        *,
        new_state: int,
        error: str | None,
        exit_code: int | None = None,
        can_retry: bool,
    ) -> None:
        """Common logic for task failure states."""
        ts = now_ms()
        task.state = new_state
        task.error = error
        task.exit_code = exit_code

        if task.attempts:
            task.attempts[-1].state = new_state
            task.attempts[-1].error = error
            task.attempts[-1].exit_code = exit_code
            task.attempts[-1].finished_at_ms = ts

        if can_retry:
            self._requeue_task(task, job, txn)
        else:
            task.finished_at_ms = ts

        self._finalize_task(task, job, txn)

    def _finalize_task(self, task: ControllerTask, job: ControllerJob, txn: TransactionLog) -> None:
        """Unassign from worker and clean up endpoints."""
        worker_id = task.worker_id
        if worker_id:
            worker = self._workers.get(worker_id)
            if worker and task.task_id in worker.running_tasks:
                worker.unassign_task(task.task_id, job.request.resources)
                txn.log("task_unassigned", task.task_id, worker_id=str(worker_id))

        self._remove_endpoints_for_task(task.task_id)

    def _requeue_task(self, task: ControllerTask, job: ControllerJob, txn: TransactionLog) -> None:
        """Put task back on scheduling queue for retry."""
        if task.task_id not in self._task_queue:
            self._task_queue.append(task.task_id)
        txn.log("task_requeued", task.task_id)

    def get_transactions(self, limit: int = 100) -> list[TransactionLog]:
        """Return recent transactions for debugging."""
        with self._lock:
            return list(self._transactions)[-limit:]

    # --- Job Management ---

    def add_job(self, job: ControllerJob, tasks: list[ControllerTask] | None = None) -> list[ControllerTask]:
        """Add a job to state, automatically creating tasks from replicas.

        If tasks are not provided, they are automatically created based on
        the job's resources.replicas field (defaulting to 1). Each task gets
        a unique ID of the form "{job_id}/task-{index}".

        Args:
            job: The job to add
            tasks: Pre-created tasks (optional, primarily for testing)

        Returns:
            List of tasks associated with this job
        """
        with self._lock:
            self._jobs[job.job_id] = job
            self._tasks_by_job[job.job_id] = []

            if tasks is None:
                tasks = expand_job_to_tasks(job)

            # Initialize task state counts
            job.num_tasks = len(tasks)
            for task in tasks:
                self._tasks[task.task_id] = task
                self._tasks_by_job[job.job_id].append(task.task_id)
                self._task_queue.append(task.task_id)
                job.task_state_counts[task.state] += 1

            if job.gang_id:
                self._gangs.setdefault(job.gang_id, set()).add(job.job_id)

            return tasks

    def get_job(self, job_id: JobId) -> ControllerJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all_jobs(self) -> list[ControllerJob]:
        with self._lock:
            return list(self._jobs.values())

    def get_gang_jobs(self, gang_id: str) -> list[ControllerJob]:
        with self._lock:
            job_ids = self._gangs.get(gang_id, set())
            return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    def get_children(self, job_id: JobId) -> list[ControllerJob]:
        with self._lock:
            return [job for job in self._jobs.values() if job.parent_job_id == job_id]

    # --- Task Management ---

    def get_task(self, task_id: TaskId) -> ControllerTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_job_tasks(self, job_id: JobId) -> list[ControllerTask]:
        """Get all tasks for a job."""
        with self._lock:
            task_ids = self._tasks_by_job.get(job_id, [])
            return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def peek_pending_tasks(self) -> list[ControllerTask]:
        """Return all schedulable tasks in queue order without removing them.

        A task is schedulable if it has no attempts yet (fresh task) or
        its current attempt is terminal and it should retry.
        """
        with self._lock:
            pending = []
            for task_id in self._task_queue:
                task = self._tasks.get(task_id)
                if task and task.can_be_scheduled():
                    pending.append(task)
            return pending

    def assign_task_to_worker(self, worker_id: WorkerId, task_id: TaskId) -> bool:
        """Assign a task to a worker and remove from queue.

        Updates the worker's committed resources based on the job's resource requirements.
        """
        with self._lock:
            worker = self._workers[worker_id]
            task = self._tasks[task_id]
            job = self._jobs[task.job_id]

            worker.assign_task(task_id, job.request.resources)
            self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
            return True

    def rollback_task_assignment(self, worker_id: WorkerId, task: ControllerTask) -> None:
        """Rollback a failed assignment: unassign from worker and re-queue."""
        with self._lock:
            worker = self._workers[worker_id]
            job = self._jobs[task.job_id]
            worker.unassign_task(task.task_id, job.request.resources)
            if task.task_id not in self._task_queue:
                self._task_queue.append(task.task_id)

    def mark_task_dispatched(self, task: ControllerTask, worker_id: WorkerId) -> int | None:
        """Mark a task as dispatched and update job state counts.

        Returns:
            New job state if changed, None otherwise
        """
        with self._lock:
            job = self._jobs[task.job_id]
            old_state = task.state
            task.create_attempt(worker_id)
            new_job_state = job.on_task_transition(old_state, task.state)
            if new_job_state is not None:
                job.state = new_job_state
            return new_job_state

    def revert_task_dispatch(self, task: ControllerTask) -> None:
        """Revert a failed dispatch and update job state counts."""
        with self._lock:
            job = self._jobs[task.job_id]
            old_state = task.state
            task.revert_attempt()
            job.task_state_counts[old_state] -= 1
            job.task_state_counts[task.state] += 1

    def transition_task(
        self,
        task_id: TaskId,
        new_state: int,
        *,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> tuple[TaskTransitionResult, list[ControllerEndpoint]]:
        """Transition a task and handle all side effects automatically.

        Whether this is a worker failure is derived from the target state:
        - TASK_STATE_WORKER_FAILED -> uses preemption retry budget
        - TASK_STATE_FAILED -> uses failure retry budget

        Side effects based on result:
        - SHOULD_RETRY: Re-queues task, unassigns from worker
        - Terminal state: Removes from queue, removes endpoints, unassigns from worker,
                         updates job state
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return TaskTransitionResult.COMPLETE, []

            job = self._jobs.get(task.job_id)
            if not job:
                return TaskTransitionResult.COMPLETE, []

            worker_id = task.worker_id
            old_state = task.state

            # Delegate state transition to task (handles both with/without attempts)
            result = task.handle_attempt_result(
                new_state,
                error=error,
                exit_code=exit_code,
            )

            # Update job's task state counts and check for job state change
            new_job_state = job.on_task_transition(old_state, task.state)
            removed_endpoints = self._update_task_queue(task, job, worker_id, result)

            # Finalize job state if it changed
            if new_job_state is not None:
                self._finalize_job_state(job, new_job_state)

            return result, removed_endpoints

    def _update_task_queue(
        self,
        task: ControllerTask,
        job: ControllerJob,
        worker_id: WorkerId | None,
        result: TaskTransitionResult,
    ) -> list[ControllerEndpoint]:
        """Update task queue based on transition result."""
        removed_endpoints: list[ControllerEndpoint] = []

        if result == TaskTransitionResult.SHOULD_RETRY:
            if task.task_id not in self._task_queue:
                self._task_queue.append(task.task_id)
            if worker_id:
                worker = self._workers.get(worker_id)
                if worker:
                    worker.unassign_task(task.task_id, job.request.resources)
        elif task.is_finished():
            self._task_queue = deque(tid for tid in self._task_queue if tid != task.task_id)
            removed_endpoints = self._remove_endpoints_for_task(task.task_id)
            if worker_id:
                worker = self._workers.get(worker_id)
                if worker:
                    worker.unassign_task(task.task_id, job.request.resources)

        return removed_endpoints

    def _finalize_job_state(
        self,
        job: ControllerJob,
        new_state: int,
    ) -> None:
        """Apply a job state change and handle associated actions."""
        job.state = new_state
        job.finished_at_ms = now_ms()

        if new_state == cluster_pb2.JOB_STATE_FAILED:
            job.error = self._get_first_task_error(job.job_id)
            self._kill_remaining_tasks(job.job_id, "Job exceeded max_task_failures")
        elif new_state == cluster_pb2.JOB_STATE_SUCCEEDED:
            pass
        elif new_state == cluster_pb2.JOB_STATE_KILLED:
            job.error = self._get_first_task_error(job.job_id)
            self._kill_remaining_tasks(job.job_id, "Job was terminated.")
        elif new_state == cluster_pb2.JOB_STATE_UNSCHEDULABLE:
            job.error = self._get_first_task_error(job.job_id)
            self._kill_remaining_tasks(job.job_id, "Job could not be scheduled.")

    def _get_first_task_error(self, job_id: JobId) -> str | None:
        """Get the first error message from failed/killed tasks in a job."""
        for task_id in self._tasks_by_job.get(job_id, []):
            task = self._tasks.get(task_id)
            if task and task.error:
                return task.error
        return None

    def _kill_remaining_tasks(self, job_id: JobId, error: str) -> None:
        """Kill all non-finished tasks in a job (failure domain).

        Updates job.task_state_counts incrementally for each killed task.
        """
        job = self._jobs.get(job_id)
        ts = now_ms()
        for task_id in self._tasks_by_job.get(job_id, []):
            task = self._tasks.get(task_id)
            if not task or task.is_finished():
                continue

            old_state = task.state
            task.state = cluster_pb2.TASK_STATE_KILLED
            task.finished_at_ms = ts
            task.error = error

            # Update job's task state counts
            if job:
                job.task_state_counts[old_state] -= 1
                job.task_state_counts[cluster_pb2.TASK_STATE_KILLED] += 1

            self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
            if task.worker_id and job:
                worker = self._workers.get(task.worker_id)
                if worker:
                    worker.unassign_task(task_id, job.request.resources)
            self._remove_endpoints_for_task(task_id)

    # --- Worker Management ---

    def add_worker(self, worker: ControllerWorker) -> None:
        with self._lock:
            self._workers[worker.worker_id] = worker

    def get_worker(self, worker_id: WorkerId) -> ControllerWorker | None:
        with self._lock:
            return self._workers.get(worker_id)

    def remove_worker(self, worker_id: WorkerId) -> ControllerWorker | None:
        with self._lock:
            return self._workers.pop(worker_id, None)

    def get_available_workers(self) -> list[ControllerWorker]:
        with self._lock:
            return [w for w in self._workers.values() if w.healthy]

    def list_all_workers(self) -> list[ControllerWorker]:
        with self._lock:
            return list(self._workers.values())

    def mark_worker_unhealthy(self, worker_id: WorkerId) -> ControllerWorker | None:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.healthy = False
            return worker

    def update_worker_heartbeat(
        self,
        worker_id: WorkerId,
        now_ms: int,
        metadata: cluster_pb2.WorkerMetadata | None = None,
    ) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.last_heartbeat_ms = now_ms
            worker.healthy = True
            if metadata is not None:
                worker.metadata = metadata
            return True

    def load_workers_from_config(self, configs: list[WorkerConfig]) -> None:
        """Load workers from static configuration."""
        ts = now_ms()
        for cfg in configs:
            worker = ControllerWorker(
                worker_id=WorkerId(cfg.worker_id),
                address=cfg.address,
                metadata=cfg.metadata,
                last_heartbeat_ms=ts,
            )
            self.add_worker(worker)

    # --- Endpoint Management ---

    def add_endpoint(self, endpoint: ControllerEndpoint, task_id: TaskId | None = None) -> None:
        """Add an endpoint, optionally associating it with a task."""
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            if task_id:
                self._endpoints_by_task.setdefault(task_id, set()).add(endpoint.endpoint_id)

    def remove_endpoint(self, endpoint_id: str) -> ControllerEndpoint | None:
        with self._lock:
            endpoint = self._endpoints.pop(endpoint_id, None)
            if endpoint:
                # Remove from task tracking
                for task_endpoints in self._endpoints_by_task.values():
                    task_endpoints.discard(endpoint_id)
            return endpoint

    def lookup_endpoints(self, name: str) -> list[ControllerEndpoint]:
        """Find endpoints by exact name match. Only returns endpoints for jobs in RUNNING state."""
        with self._lock:
            results = []
            for ep in self._endpoints.values():
                if ep.name != name:
                    continue
                # Only return endpoints for running jobs
                job = self._jobs.get(ep.job_id)
                if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                    results.append(ep)
            return results

    def list_endpoints_by_prefix(self, prefix: str) -> list[ControllerEndpoint]:
        """List endpoints matching a name prefix. Only returns endpoints for jobs in RUNNING state."""
        with self._lock:
            results = []
            for ep in self._endpoints.values():
                if not ep.name.startswith(prefix):
                    continue
                job = self._jobs.get(ep.job_id)
                if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                    results.append(ep)
            return results

    def list_all_endpoints(self) -> list[ControllerEndpoint]:
        """Return all registered endpoints."""
        with self._lock:
            return list(self._endpoints.values())

    def _remove_endpoints_for_task(self, task_id: TaskId) -> list[ControllerEndpoint]:
        """Remove all endpoints associated with a task."""
        endpoint_ids = list(self._endpoints_by_task.get(task_id, []))
        removed = []
        for eid in endpoint_ids:
            endpoint = self._endpoints.pop(eid, None)
            if endpoint:
                removed.append(endpoint)
        self._endpoints_by_task.pop(task_id, None)
        return removed

    def remove_endpoints_for_job(self, job_id: JobId) -> list[ControllerEndpoint]:
        """Remove all endpoints for a job by removing endpoints for all its tasks."""
        with self._lock:
            all_removed = []
            for task_id in self._tasks_by_job.get(job_id, []):
                removed = self._remove_endpoints_for_task(task_id)
                all_removed.extend(removed)
            return all_removed
