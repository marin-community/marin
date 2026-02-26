# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

import bisect
import logging
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import NamedTuple

from iris.cluster.controller.events import (
    Event,
    JobCancelledEvent,
    JobSubmittedEvent,
    TaskAssignedEvent,
    TaskStateChangedEvent,
    TransactionLog,
    WorkerFailedEvent,
    WorkerHeartbeatEvent,
    WorkerHeartbeatFailedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.types import (
    AttributeValue,
    JobName,
    WorkerId,
    get_device_type,
    get_device_variant,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import cluster_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)

MAX_REPLICAS_PER_JOB = 10000
"""Maximum replicas allowed per job to prevent resource exhaustion."""

DEFAULT_MAX_RETRIES_PREEMPTION = 100
"""Default preemption retries. High because worker failures are typically transient."""

HEARTBEAT_FAILURE_THRESHOLD = 10
"""Consecutive heartbeat failures before marking worker as failed."""


# =============================================================================
# Device Helper Functions
# =============================================================================


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

# Terminal states that originate from worker reports via heartbeat (as opposed to
# controller decisions like KILLED or UNSCHEDULABLE). Used to detect duplicate
# completions across multiple heartbeats.
WORKER_REPORTED_TERMINAL_STATES: frozenset[int] = frozenset(
    {
        cluster_pb2.TASK_STATE_SUCCEEDED,
        cluster_pb2.TASK_STATE_FAILED,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
    }
)


class TaskTransitionResult(Enum):
    """Result of a task state transition.

    APPLIED: Transition processed. No special caller action needed.
    SHOULD_RETRY: Task failed but has retry budget. Caller should requeue.
    EXCEEDED_RETRY_LIMIT: Task failed and exhausted retries. Terminal.
    """

    APPLIED = "applied"
    SHOULD_RETRY = "should_retry"
    EXCEEDED_RETRY_LIMIT = "exceeded_retry_limit"


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
    state: int = cluster_pb2.TASK_STATE_ASSIGNED
    log_directory: str | None = None  # Storage location for logs

    # Timing
    created_at: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None

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
        now = Timestamp.now()

        if new_state == cluster_pb2.TASK_STATE_RUNNING:
            self.started_at = now

        if new_state in TERMINAL_TASK_STATES:
            self.finished_at = now
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

    State Diagram::

                            +-----------+
                            |  PENDING  |<-----------------+
                            +-----+-----+                  |
                                  |                        |
                            dispatch to worker             |
                                  |                        |
                                  v                        |
                            +-----------+                  |
                            | ASSIGNED  |                  |
                            +-----+-----+                  |
                                  |                        |
                          worker starts task               |
                                  |                        |
                                  v                        |
                            +-----------+                  |
                            |  RUNNING  |                  |
                            +-----+-----+                  |
                                  |                        |
              +-------------------+-------------------+    |
              |                   |                   |    |
              v                   v                   v    |
        +-----------+       +-----------+      +-----------+
        | SUCCEEDED |       |  FAILED   |----->| (retry)   |
        +-----------+       +-----------+      +-----------+
                                  |                  ^
                                  | exhausted        |
                                  v                  |
                            (terminal)               |
                                                     |
                            +-----------+            |
                            |WORKER_FAIL|------------+
                            +-----------+
                                  |
                                  | exhausted
                                  v
                            (terminal)

        Other terminal states: KILLED, UNSCHEDULABLE (no retry)

    Retry Semantics:
        - failure_count: Incremented on TASK_STATE_FAILED. Checked against max_retries_failure.
        - preemption_count: Incremented on TASK_STATE_WORKER_FAILED. Checked against max_retries_preemption.

        When a task fails with retry budget remaining, it stays in the failure state but
        can_be_scheduled() returns True. The scheduler creates a new attempt when re-dispatching.

    State transitions are handled via handle_attempt_result(), which updates both
    the attempt and task-level state, returning a TaskTransitionResult indicating
    what the caller should do.
    """

    task_id: JobName
    job_id: JobName

    # Task owns its state directly
    state: int = cluster_pb2.TASK_STATE_PENDING
    error: str | None = None
    exit_code: int | None = None
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None

    # Retry policy (immutable after creation)
    max_retries_failure: int = 0
    max_retries_preemption: int = DEFAULT_MAX_RETRIES_PREEMPTION

    # Retry counters (task-level, not attempt-level)
    failure_count: int = 0
    preemption_count: int = 0

    # Attempt tracking - pure execution history
    attempts: list[ControllerTaskAttempt] = field(default_factory=list)

    # Submission timestamp (distinct from attempt start times)
    submitted_at: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))

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
    def task_index(self) -> int:
        """0-indexed task number within the job."""
        return self.task_id.require_task()[1]

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

    def create_attempt(
        self,
        worker_id: WorkerId,
        *,
        initial_state: int = cluster_pb2.TASK_STATE_ASSIGNED,
    ) -> ControllerTaskAttempt:
        """Create a new attempt for this task.

        Called when the scheduler assigns this task to a worker. The attempt
        starts in ASSIGNED state by default - the worker will report RUNNING
        when execution actually begins.

        Args:
            worker_id: The worker this attempt is assigned to
            initial_state: Starting state for the attempt (default: ASSIGNED)

        Returns:
            The new attempt so caller can track it.
        """
        attempt = ControllerTaskAttempt(
            attempt_id=len(self.attempts),
            worker_id=worker_id,
            state=initial_state,
            created_at=Timestamp.now(),
        )
        self.attempts.append(attempt)

        # Update task-level state
        self.state = initial_state

        return attempt

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
        - If terminal success: returns APPLIED, task state is terminal
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
                self.finished_at = Timestamp.now()
                self.error = error
                self.exit_code = exit_code
            return TaskTransitionResult.APPLIED

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
                self.finished_at = Timestamp.now()
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
            self.finished_at = Timestamp.now()
            return TaskTransitionResult.APPLIED

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
            self.finished_at = Timestamp.now()
            return TaskTransitionResult.APPLIED

        # Non-terminal states (BUILDING, RUNNING)
        attempt.transition(
            new_state,
            exit_code=exit_code,
            error=error,
        )
        # Update task-level state for non-terminal transitions
        self.state = new_state
        return TaskTransitionResult.APPLIED

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
        - It has no attempts yet (fresh task) AND is not in a terminal state, or
        - Its current attempt is terminal AND it should retry
        """
        if self.state in TERMINAL_TASK_STATES:
            return False
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
    """Controller representation of a job.

    Job state is derived from task-state counts via on_task_transition()
    and _compute_job_state(); it is not managed by a standalone job.transition()
    retry state machine.
    """

    job_id: JobName
    request: cluster_pb2.Controller.LaunchJobRequest
    state: int = cluster_pb2.JOB_STATE_PENDING

    # Timestamps
    submitted_at: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    root_submitted_at: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None

    # Scheduling deadline (set when job has a scheduling_timeout)
    scheduling_deadline: Deadline | None = None

    error: str | None = None
    exit_code: int | None = None

    # Incremental task state tracking
    num_tasks: int = 0
    task_state_counts: Counter[int] = field(default_factory=Counter)

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

        # Job is RUNNING if any task is assigned, building, or running
        if (
            counts[cluster_pb2.TASK_STATE_RUNNING] > 0
            or counts[cluster_pb2.TASK_STATE_BUILDING] > 0
            or counts[cluster_pb2.TASK_STATE_ASSIGNED] > 0
        ) and self.state != cluster_pb2.JOB_STATE_RUNNING:
            if self.started_at is None:
                self.started_at = Timestamp.now()
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

    @property
    def is_coscheduled(self) -> bool:
        """Whether this job uses coscheduling (all tasks assigned atomically)."""
        return self.request.HasField("coscheduling")

    @property
    def coscheduling_group_by(self) -> str | None:
        """The attribute key used to group workers for coscheduling, or None."""
        if self.is_coscheduled:
            return self.request.coscheduling.group_by
        return None


# =============================================================================
# Job Helper Functions
# =============================================================================


def expand_job_to_tasks(job: ControllerJob) -> list[ControllerTask]:
    """Expand a job into its constituent tasks based on replicas.

    Jobs with replicas=N expand into N tasks. Each task has a unique ID
    of the form "/job/.../index" where index is 0-based.

    Args:
        job: The job to expand

    Returns:
        List of ControllerTask objects for this job

    Raises:
        ValueError: If replicas is < 1 or exceeds MAX_REPLICAS_PER_JOB
    """
    num_replicas = job.request.replicas

    if num_replicas < 1:
        raise ValueError(f"Job {job.job_id} has invalid replicas={num_replicas}; must be >= 1")
    if num_replicas > MAX_REPLICAS_PER_JOB:
        raise ValueError(f"Job {job.job_id} replicas={num_replicas} exceeds max {MAX_REPLICAS_PER_JOB}")

    tasks = []

    for i in range(num_replicas):
        task_id = job.job_id.task(i)
        task = ControllerTask(
            task_id=task_id,
            job_id=job.job_id,
            max_retries_failure=job.request.max_retries_failure,
            max_retries_preemption=job.request.max_retries_preemption,
            submitted_at=job.submitted_at,
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
        last_heartbeat: Timestamp of last successful heartbeat
        running_tasks: Set of task IDs currently running on this worker
        committed_cpu_millicores: Total CPU millicores committed to running tasks
        committed_mem: Total memory bytes committed to running tasks
        committed_gpu: Total GPUs committed to running tasks
        attributes: Typed attributes for constraint-based scheduling (e.g., tpu-name, tpu-worker-id)
    """

    worker_id: WorkerId
    address: str
    metadata: cluster_pb2.WorkerMetadata

    # Health tracking
    healthy: bool = True
    consecutive_failures: int = 0
    last_heartbeat: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))

    # Currently running tasks
    running_tasks: set[JobName] = field(default_factory=set)

    # Committed resources (tracked incrementally)
    committed_cpu_millicores: int = 0
    committed_mem: int = 0
    committed_gpu: int = 0
    committed_tpu: int = 0

    # Worker attributes for constraint-based scheduling
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

    # All task IDs ever assigned to this worker (for fast per-worker history lookup)
    task_history: set[JobName] = field(default_factory=set)

    def get_committed_resources(self) -> tuple[int, int, int]:
        """Return committed (cpu_millicores, memory_bytes, gpu_count) for this worker."""
        return (self.committed_cpu_millicores, self.committed_mem, self.committed_gpu)

    def heartbeat_deadline(self, timeout: Duration) -> Timestamp:
        """Compute the deadline when this worker's heartbeat will expire.

        Args:
            timeout: Heartbeat timeout duration

        Returns:
            Timestamp when the heartbeat will be considered expired
        """
        return self.last_heartbeat.add(timeout)

    def is_heartbeat_expired(self, timeout: Duration) -> bool:
        """Check if this worker's heartbeat has expired.

        Args:
            timeout: Heartbeat timeout duration

        Returns:
            True if the worker has not sent a heartbeat within the timeout period
        """
        return self.last_heartbeat.age_ms() > timeout.to_ms()

    def assign_task(self, task_id: JobName, resources: cluster_pb2.ResourceSpecProto) -> None:
        """Assign a task to this worker, updating committed resources."""
        self.running_tasks.add(task_id)
        self.task_history.add(task_id)
        self.committed_cpu_millicores += resources.cpu_millicores
        self.committed_mem += resources.memory_bytes
        self.committed_gpu += get_gpu_count(resources.device)
        self.committed_tpu += get_tpu_count(resources.device)

    def unassign_task(self, task_id: JobName, resources: cluster_pb2.ResourceSpecProto) -> None:
        """Unassign a task from this worker, updating committed resources."""
        self.running_tasks.discard(task_id)
        self.committed_cpu_millicores -= resources.cpu_millicores
        self.committed_mem -= resources.memory_bytes
        self.committed_gpu -= get_gpu_count(resources.device)
        self.committed_tpu -= get_tpu_count(resources.device)

    @property
    def available_cpu_millicores(self) -> int:
        """Available CPU millicores after subtracting committed resources."""
        return self.metadata.cpu_count * 1000 - self.committed_cpu_millicores

    @property
    def available_memory(self) -> int:
        """Available memory bytes after subtracting committed resources."""
        return self.metadata.memory_bytes - self.committed_mem

    @property
    def available_gpus(self) -> int:
        """Available GPU count after subtracting committed resources."""
        return get_gpu_count(self.metadata.device) - self.committed_gpu

    @property
    def available_tpus(self) -> int:
        """Available TPU chip count after subtracting committed resources."""
        return get_tpu_count(self.metadata.device) - self.committed_tpu

    @property
    def device_type(self) -> str:
        """Device type from worker metadata."""
        return get_device_type(self.metadata.device)

    @property
    def device_variant(self) -> str | None:
        """Device variant from worker metadata."""
        return get_device_variant(self.metadata.device)


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
        registered_at: Timestamp when endpoint was registered
    """

    endpoint_id: str
    name: str  # Full prefixed name: "{root_job_id}/{actor_name}"
    address: str
    job_id: JobName
    metadata: dict[str, str] = field(default_factory=dict)
    registered_at: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))


# =============================================================================
# Heartbeat Dispatch Types
# =============================================================================


class TaskPriorityKey(NamedTuple):
    """Priority key for depth-first task ordering.

    Lower values = higher priority.
    - neg_depth: Negative depth (deeper jobs sort first)
    - root_submitted_ms: Root job submission time (older trees first)
    - submitted_ms: Task submission time (FIFO within tree)
    """

    neg_depth: int
    root_submitted_ms: int
    submitted_ms: int


class QueueEntry(NamedTuple):
    """Entry in the priority-sorted task queue.

    - priority: TaskPriorityKey for ordering
    - insertion_order: Counter to prevent comparing JobName objects
    - task_id: The task identifier
    """

    priority: TaskPriorityKey
    insertion_order: int
    task_id: JobName


@dataclass
class PendingDispatch:
    """Buffered dispatches waiting for next heartbeat.

    These are accumulated by the scheduling thread and drained atomically
    when preparing a heartbeat snapshot.
    """

    tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest] = field(default_factory=list)
    tasks_to_kill: list[str] = field(default_factory=list)


class RunningTaskEntry(NamedTuple):
    """Task ID and attempt ID pair captured at snapshot time."""

    task_id: JobName
    attempt_id: int


@dataclass
class HeartbeatSnapshot:
    """Immutable snapshot of worker state for heartbeat dispatch.

    Captures all data needed to send a heartbeat RPC without holding locks:
    - Worker identity and address for RPC routing
    - Running tasks with attempt IDs for reconciliation
    - Buffered dispatches/kills to deliver

    Taken atomically under state lock to prevent iteration races.
    """

    worker_id: WorkerId
    worker_address: str
    vm_address: str
    running_tasks: list[RunningTaskEntry]
    tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest]
    tasks_to_kill: list[str]


# =============================================================================
# Controller State
# =============================================================================


class ControllerState:
    """Thread-safe controller state managing jobs, tasks, workers, endpoints, and queues.

    API Design:

    1. Event API (Observable State Transitions):
       - handle_event() - Main entry point for all observable state changes
       - Use for: external triggers, worker reports, system events
       - Returns TransactionLog for debugging

    2. Query API (Read-Only Access):
       - peek_pending_tasks(), get_available_workers(), get_job(), etc.
       - Used by scheduler and controller for reads

    3. Test Utilities (Setup Helpers):
       - add_job(), add_worker() - Direct object creation for tests
       - Bypass event logs for test convenience

    State Commitment Flow:
       The scheduler returns pure assignment pairs without mutating state.
       State is only committed after successful dispatch RPC via TASK_ASSIGNED event,
       which creates the attempt and commits resources.
    """

    def __init__(self, heartbeat_failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD):
        self._lock = RLock()
        self._heartbeat_failure_threshold = heartbeat_failure_threshold
        self._jobs: dict[JobName, ControllerJob] = {}
        self._tasks: dict[JobName, ControllerTask] = {}
        self._tasks_by_job: dict[JobName, list[JobName]] = {}
        self._workers: dict[WorkerId, ControllerWorker] = {}
        # Priority-sorted task queue. Sorted ascending — lower keys = higher priority.
        self._task_queue: list[QueueEntry] = []
        self._endpoints: dict[str, ControllerEndpoint] = {}  # endpoint_id -> endpoint
        self._endpoints_by_task: dict[JobName, set[str]] = {}  # task_id -> endpoint_ids
        self._transactions: deque[TransactionLog] = deque(maxlen=1000)  # Event transaction log
        self._pending_dispatch: dict[WorkerId, PendingDispatch] = {}  # Buffered heartbeat dispatches

    # =========================================================================
    # Event-Driven State Transitions
    # =========================================================================

    def handle_event(self, event: Event) -> TransactionLog:
        """Main entry point for all event-driven state changes.

        Dispatches to the appropriate handler based on event type and records
        all actions to a transaction log for debugging.

        Args:
            event: The event to process

        Returns:
            TransactionLog containing all actions taken during event handling
        """
        with self._lock:
            txn = TransactionLog(event=event)

            match event:
                case WorkerRegisteredEvent():
                    self._on_worker_registered(txn, event)
                case WorkerHeartbeatEvent():
                    self._on_worker_heartbeat(txn, event)
                case WorkerHeartbeatFailedEvent():
                    self._on_worker_heartbeat_failed(txn, event)
                case WorkerFailedEvent():
                    self._on_worker_failed(txn, event)
                case JobSubmittedEvent():
                    self._on_job_submitted(txn, event)
                case JobCancelledEvent():
                    self._on_job_cancelled(txn, event)
                case TaskAssignedEvent():
                    self._on_task_assigned(txn, event)
                case TaskStateChangedEvent():
                    self._on_task_state_changed(txn, event)
                case _:
                    raise TypeError(f"Unhandled event type: {type(event).__name__}")

            self._transactions.append(txn)

            # Log transaction for debugging; demote all heartbeat events to DEBUG
            if txn.actions:
                log = logger.debug if isinstance(event, WorkerHeartbeatEvent) else logger.info
                log(f"Event {type(event).__name__}: {len(txn.actions)} actions")
                for action in txn.actions:
                    log(f"  - {action.action} {action.entity_id} {action.details}")

            return txn

    # -------------------------------------------------------------------------
    # Worker Event Handlers
    # -------------------------------------------------------------------------

    def _on_worker_registered(self, txn: TransactionLog, event: WorkerRegisteredEvent) -> None:
        worker = self._workers.get(event.worker_id)

        # Extract attributes from metadata proto
        attributes = {k: AttributeValue.from_proto(v) for k, v in event.metadata.attributes.items()}

        if worker:
            # Existing worker - heartbeat update
            worker.last_heartbeat = event.timestamp
            worker.healthy = True
            worker.consecutive_failures = 0
            worker.metadata = event.metadata
            worker.attributes = attributes
            txn.log("worker_heartbeat", event.worker_id)
        else:
            # New worker - full registration
            worker = ControllerWorker(
                worker_id=event.worker_id,
                address=event.address,
                metadata=event.metadata,
                last_heartbeat=event.timestamp,
                attributes=attributes,
            )
            self._workers[event.worker_id] = worker
            txn.log("worker_registered", event.worker_id, address=event.address)
        logger.info(
            "Worker registered: id=%s addr=%s gpu_count=%d gpu_name=%s attrs=%s",
            event.worker_id,
            event.address,
            event.metadata.gpu_count,
            event.metadata.gpu_name,
            sorted(attributes.keys()),
        )

    def _on_worker_heartbeat(self, txn: TransactionLog, event: WorkerHeartbeatEvent) -> None:
        worker = self._workers[event.worker_id]
        worker.last_heartbeat = event.timestamp
        worker.healthy = True
        worker.consecutive_failures = 0
        txn.log("heartbeat", event.worker_id)

    def _on_worker_heartbeat_failed(self, txn: TransactionLog, event: WorkerHeartbeatFailedEvent) -> None:
        worker = self._workers.get(event.worker_id)
        if not worker:
            return
        worker.consecutive_failures += 1
        txn.log("heartbeat_failed", event.worker_id, consecutive=worker.consecutive_failures)
        if worker.consecutive_failures >= self._heartbeat_failure_threshold:
            logger.warning(
                "Worker %s exceeded heartbeat failure threshold: consecutive_failures=%d threshold=%d error=%s",
                event.worker_id,
                worker.consecutive_failures,
                self._heartbeat_failure_threshold,
                event.error,
            )
            self._on_worker_failed(txn, WorkerFailedEvent(worker_id=event.worker_id, error=event.error))

    def _on_worker_failed(self, txn: TransactionLog, event: WorkerFailedEvent) -> None:
        worker = self._workers[event.worker_id]
        worker.healthy = False

        affected_task_ids = list(worker.running_tasks)
        logger.warning(
            "Worker %s failed: error=%s affected_tasks=%d task_ids=%s",
            event.worker_id,
            event.error,
            len(affected_task_ids),
            [str(tid) for tid in affected_task_ids],
        )
        txn.log(
            "worker_failed",
            event.worker_id,
            error=event.error,
            affected_task_count=len(affected_task_ids),
            affected_task_ids=[str(tid) for tid in affected_task_ids],
        )

        # Cascade to all tasks on this worker (RUNNING, ASSIGNED, or BUILDING) - call handler directly, same transaction
        for task_id in affected_task_ids:
            task = self._tasks[task_id]
            assert task.worker_id == event.worker_id
            # Skip terminal states
            if task.state in TERMINAL_TASK_STATES:
                continue

            cascade_event = TaskStateChangedEvent(
                task_id=task_id,
                new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
                attempt_id=task.current_attempt_id,
                error=f"Worker {event.worker_id} failed: {event.error or 'unknown'}",
            )
            self._on_task_state_changed(txn, cascade_event)

        # Prune the dead worker from state so it doesn't accumulate in the
        # controller list and confuse the dashboard. If the worker comes back,
        # it will re-register as a fresh entry.
        del self._workers[event.worker_id]
        self._pending_dispatch.pop(event.worker_id, None)
        txn.log("worker_pruned", event.worker_id)

    # -------------------------------------------------------------------------
    # Job Event Handlers
    # -------------------------------------------------------------------------

    def _on_job_submitted(self, txn: TransactionLog, event: JobSubmittedEvent) -> None:
        job = ControllerJob(
            job_id=event.job_id,
            request=event.request,
            submitted_at=event.timestamp,
        )
        if job.request.HasField("scheduling_timeout") and job.request.scheduling_timeout.milliseconds > 0:
            job.scheduling_deadline = Deadline.from_now(Duration.from_proto(job.request.scheduling_timeout))

        # Resolve root submission time for depth-first priority ordering.
        # Child jobs inherit the root timestamp so the entire tree sorts together.
        if event.job_id.is_root:
            job.root_submitted_at = event.timestamp
        else:
            parent_job = self._jobs.get(event.job_id.parent)
            if parent_job:
                job.root_submitted_at = parent_job.root_submitted_at
            else:
                # Orphan child (parent already cleaned up) — treat as new root.
                job.root_submitted_at = event.timestamp

        self._jobs[event.job_id] = job
        self._tasks_by_job[event.job_id] = []

        try:
            tasks = expand_job_to_tasks(job)
        except ValueError as e:
            job.state = cluster_pb2.JOB_STATE_FAILED
            job.error = str(e)
            job.finished_at = Timestamp.now()
            txn.log("job_validation_failed", event.job_id, error=str(e))
            return

        job.num_tasks = len(tasks)

        for task in tasks:
            self._tasks[task.task_id] = task
            self._tasks_by_job[event.job_id].append(task.task_id)
            self._enqueue_task(task.task_id)
            job.task_state_counts[task.state] += 1
            txn.log("task_created", task.task_id, job_id=str(event.job_id))

        txn.log("job_submitted", event.job_id, num_tasks=job.num_tasks)

    def _on_job_cancelled(self, txn: TransactionLog, event: JobCancelledEvent) -> None:
        job = self._jobs[event.job_id]

        for task_id in self._tasks_by_job.get(event.job_id, []):
            task = self._tasks[task_id]
            if task.state in TERMINAL_TASK_STATES:
                continue

            # Track running tasks for kill RPC. Unlike task failures that trigger
            # _finalize_job_state (which populates tasks_to_kill), job cancellation
            # sets job state directly, so we must track kill targets here.
            if task.worker_id:
                txn.tasks_to_kill.add(task_id)

            cascade_event = TaskStateChangedEvent(
                task_id=task_id,
                new_state=cluster_pb2.TASK_STATE_KILLED,
                attempt_id=task.current_attempt_id,
                error=event.reason,
            )
            self._on_task_state_changed(txn, cascade_event)

        job.state = cluster_pb2.JOB_STATE_KILLED
        job.error = event.reason
        job.finished_at = Timestamp.now()
        txn.log("job_cancelled", event.job_id, reason=event.reason)

    # -------------------------------------------------------------------------
    # Task Event Handlers
    # -------------------------------------------------------------------------

    def _on_task_assigned(self, txn: TransactionLog, event: TaskAssignedEvent) -> None:
        """Handle successful task assignment to a worker.

        Called AFTER successful dispatch RPC. Commits resources and creates attempt.

        Creates an attempt in ASSIGNED state. The task transitions to RUNNING
        when the worker reports TASK_STATE_CHANGED with RUNNING.
        """
        task = self._tasks[event.task_id]
        job = self._jobs[task.job_id]
        worker = self._workers[event.worker_id]

        old_state = task.state
        task.create_attempt(event.worker_id, initial_state=cluster_pb2.TASK_STATE_ASSIGNED)

        # Commit resources and add to running_tasks
        worker.assign_task(event.task_id, job.request.resources)

        # Update job counters (state stays PENDING, so no job state change expected)
        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            job.state = new_job_state

        txn.log("task_assigned", event.task_id, worker_id=str(event.worker_id))

    def _on_task_state_changed(self, txn: TransactionLog, event: TaskStateChangedEvent) -> None:
        """Handle all task state transitions.

        Delegates to task.handle_attempt_result() which contains the canonical
        state transition logic including retry budget management.
        """
        task = self._tasks[event.task_id]

        # Validate attempt_id matches the current attempt
        if event.attempt_id != task.current_attempt_id:
            stale_attempt = task.attempts[event.attempt_id] if 0 <= event.attempt_id < len(task.attempts) else None
            if stale_attempt and not stale_attempt.is_terminal():
                logger.error(
                    "Stale attempt precondition violation: task=%s received attempt=%d "
                    "but current is %d and stale attempt state is %s (not terminal)",
                    event.task_id,
                    event.attempt_id,
                    task.current_attempt_id,
                    stale_attempt.state,
                )
            else:
                logger.warning(
                    "Ignoring stale task state report: task=%s attempt=%d current=%d",
                    event.task_id,
                    event.attempt_id,
                    task.current_attempt_id,
                )
            txn.log(
                "stale_attempt_ignored",
                event.task_id,
                reported_attempt=event.attempt_id,
                current_attempt=task.current_attempt_id,
            )
            return

        # Ignore duplicate worker-reported completions for an attempt already processed.
        # This can happen when a heartbeat response is retried (e.g., the previous
        # heartbeat succeeded on the controller but the worker didn't get the ack).
        # Only guard worker-reported states (FAILED, WORKER_FAILED, SUCCEEDED);
        # controller-originated states (UNSCHEDULABLE, KILLED) are always allowed.
        current_attempt = task.current_attempt
        if (
            current_attempt
            and current_attempt.is_terminal()
            and task.state == cluster_pb2.TASK_STATE_PENDING
            and event.new_state in WORKER_REPORTED_TERMINAL_STATES
        ):
            logger.debug(
                "Ignoring duplicate terminal report for requeued task: task=%s attempt=%d "
                "attempt_state=%s, task already requeued to PENDING",
                event.task_id,
                event.attempt_id,
                current_attempt.state,
            )
            txn.log(
                "duplicate_terminal_ignored",
                event.task_id,
                attempt=event.attempt_id,
                existing_state=current_attempt.state,
            )
            return

        job = self._jobs[task.job_id]
        old_state = task.state

        # Delegate to the canonical state transition logic
        result = task.handle_attempt_result(
            event.new_state,
            error=event.error,
            exit_code=event.exit_code,
        )

        # Handle side effects based on result
        is_finished = task.is_finished()
        if result == TaskTransitionResult.SHOULD_RETRY:
            self._requeue_task(task, txn)
            self._cleanup_task_resources(task, job, txn)
        elif is_finished:
            self._cleanup_task_resources(task, job, txn)

        # Log task state change with retry context for debugging preemption issues
        old_state_name = cluster_pb2.TaskState.Name(old_state)
        new_state_name = cluster_pb2.TaskState.Name(task.state)

        if result == TaskTransitionResult.EXCEEDED_RETRY_LIMIT:
            logger.warning(
                "Task %s exhausted retries: %s -> %s attempt=%d "
                "failure_count=%d/%d preemption_count=%d/%d result=%s is_finished=%s",
                event.task_id,
                old_state_name,
                new_state_name,
                event.attempt_id,
                task.failure_count,
                task.max_retries_failure,
                task.preemption_count,
                task.max_retries_preemption,
                result.name,
                is_finished,
            )
        elif event.new_state == cluster_pb2.TASK_STATE_WORKER_FAILED:
            logger.warning(
                "Task %s worker failed: %s -> %s attempt=%d "
                "failure_count=%d/%d preemption_count=%d/%d result=%s is_finished=%s",
                event.task_id,
                old_state_name,
                new_state_name,
                event.attempt_id,
                task.failure_count,
                task.max_retries_failure,
                task.preemption_count,
                task.max_retries_preemption,
                result.name,
                is_finished,
            )
        else:
            logger.info(
                "Task %s state changed: %s -> %s attempt=%d "
                "failure_count=%d/%d preemption_count=%d/%d result=%s is_finished=%s",
                event.task_id,
                old_state_name,
                new_state_name,
                event.attempt_id,
                task.failure_count,
                task.max_retries_failure,
                task.preemption_count,
                task.max_retries_preemption,
                result.name,
                is_finished,
            )

        # Coscheduled group failure: if one task fails terminally, kill all running siblings.
        # For multi-host TPU jobs, if one host fails the other hosts cannot continue
        # (collective ops will timeout), so we immediately kill all running siblings.
        if (
            job.is_coscheduled
            and task.is_finished()
            and task.state
            in (
                cluster_pb2.TASK_STATE_FAILED,
                cluster_pb2.TASK_STATE_WORKER_FAILED,
            )
        ):
            self._cascade_coscheduled_failure(task, job, txn)

        # Update job state counters and finalize if needed
        new_job_state = job.on_task_transition(old_state, task.state)
        if new_job_state is not None:
            killed_tasks = self._finalize_job_state(job, new_job_state, txn)
            txn.tasks_to_kill.update(killed_tasks)

        txn.log(
            "task_state_changed",
            event.task_id,
            old_state=old_state_name,
            new_state=new_state_name,
            attempt_id=event.attempt_id,
            failure_count=task.failure_count,
            max_retries_failure=task.max_retries_failure,
            preemption_count=task.preemption_count,
            max_retries_preemption=task.max_retries_preemption,
            result=result.name,
            is_finished=is_finished,
        )

    # -------------------------------------------------------------------------
    # Shared Helpers for Event Handlers
    # -------------------------------------------------------------------------

    def _task_priority_key(self, task_id: JobName) -> TaskPriorityKey:
        """Priority key for depth-first task ordering. Lower values = higher priority.

        Deeper jobs sort first (negative depth). Among same depth, older root
        trees are preferred. Within the same tree and depth, FIFO by submission.
        """
        task = self._tasks.get(task_id)
        job = self._jobs.get(task.job_id) if task else None
        if not task or not job:
            return TaskPriorityKey(0, 0, 0)
        return TaskPriorityKey(
            neg_depth=-task.job_id.depth,
            root_submitted_ms=job.root_submitted_at.epoch_ms(),
            submitted_ms=task.submitted_at.epoch_ms(),
        )

    def _enqueue_task(self, task_id: JobName) -> None:
        """Insert task into priority-sorted queue."""
        key = self._task_priority_key(task_id)
        entry = QueueEntry(priority=key, insertion_order=len(self._task_queue), task_id=task_id)
        bisect.insort(self._task_queue, entry)

    def _cleanup_task_resources(self, task: ControllerTask, job: ControllerJob, txn: TransactionLog) -> None:
        """Release worker resources and remove task endpoints."""
        worker_id = task.worker_id
        if worker_id:
            worker = self._workers.get(worker_id)
            if worker and task.task_id in worker.running_tasks:
                worker.unassign_task(task.task_id, job.request.resources)
                txn.log("task_unassigned", task.task_id, worker_id=str(worker_id))

        self._remove_endpoints_for_task(task.task_id)

    def _requeue_task(self, task: ControllerTask, txn: TransactionLog) -> None:
        """Put task back on scheduling queue for retry."""
        task.state = cluster_pb2.TASK_STATE_PENDING
        if not any(entry.task_id == task.task_id for entry in self._task_queue):
            self._enqueue_task(task.task_id)
        txn.log("task_requeued", task.task_id)

    def _cascade_coscheduled_failure(
        self,
        trigger_task: ControllerTask,
        job: ControllerJob,
        txn: TransactionLog,
    ) -> None:
        """Kill all running siblings when a coscheduled task fails terminally.

        For multi-host TPU jobs, if one host fails the other hosts cannot continue
        because collective operations will timeout. We mark siblings terminal and
        release their worker resources in one pass.
        """
        for sibling_id in self._tasks_by_job.get(job.job_id, []):
            if sibling_id == trigger_task.task_id:
                continue

            sibling = self._tasks[sibling_id]
            if sibling.state not in (cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_ASSIGNED):
                continue

            sibling_old = sibling.state

            # Exhaust preemption budget so that after handle_attempt_result
            # increments the counter, the sibling is terminal.
            sibling.preemption_count = sibling.max_retries_preemption

            sibling.handle_attempt_result(
                cluster_pb2.TASK_STATE_WORKER_FAILED,
                error=f"Coscheduled sibling {trigger_task.task_id} failed",
            )
            job.on_task_transition(sibling_old, sibling.state)
            self._cleanup_task_resources(sibling, job, txn)
            txn.tasks_to_kill.add(sibling_id)
            txn.log(
                "coscheduled_sibling_killed",
                sibling_id,
                trigger_task=str(trigger_task.task_id),
            )

    def get_transactions(self, limit: int = 100) -> list[TransactionLog]:
        """Return recent transactions for debugging."""
        with self._lock:
            return list(self._transactions)[-limit:]

    # --- Job Management ---
    def add_job(self, job: ControllerJob, tasks: list[ControllerTask] | None = None) -> list[ControllerTask]:
        """Add job directly to state (test utility).

        For production use, create Event(JOB_SUBMITTED) instead.
        This method bypasses event logging and is intended for test setup only.

        If tasks are not provided, they are automatically created based on
        the job's replicas field (defaulting to 1). Each task gets
        a unique ID of the form "/job/.../index".

        Args:
            job: The job to add
            tasks: Pre-created tasks (optional, primarily for testing)

        Returns:
            List of tasks associated with this job
        """
        with self._lock:
            # Resolve root_submitted_at if not already set (test helper path).
            # Match production logic from _on_job_submitted.
            if job.root_submitted_at.epoch_ms() == 0:
                if job.job_id.is_root:
                    job.root_submitted_at = job.submitted_at
                else:
                    parent_job = self._jobs.get(job.job_id.parent)
                    if parent_job:
                        job.root_submitted_at = parent_job.root_submitted_at
                    else:
                        # Orphan child (parent already cleaned up) — treat as new root.
                        job.root_submitted_at = job.submitted_at

            self._jobs[job.job_id] = job
            self._tasks_by_job[job.job_id] = []

            if tasks is None:
                tasks = expand_job_to_tasks(job)

            # Initialize task state counts
            job.num_tasks = len(tasks)
            for task in tasks:
                self._tasks[task.task_id] = task
                self._tasks_by_job[job.job_id].append(task.task_id)
                self._enqueue_task(task.task_id)
                job.task_state_counts[task.state] += 1

            return tasks

    def get_job(self, job_id: JobName) -> ControllerJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all_jobs(self) -> list[ControllerJob]:
        with self._lock:
            return list(self._jobs.values())

    def get_children(self, job_id: JobName) -> list[ControllerJob]:
        with self._lock:
            return [job for job in self._jobs.values() if job.job_id.parent == job_id]

    def remove_finished_job(self, job_id: JobName) -> bool:
        """Remove a finished job and its tasks from state.

        Only removes jobs that are in a terminal state (SUCCEEDED, FAILED, KILLED,
        UNSCHEDULABLE). This allows job names to be reused after completion.

        Args:
            job_id: The job ID to remove

        Returns:
            True if the job was removed, False if it doesn't exist or is not finished
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False

            if not job.is_finished():
                return False

            # Remove all tasks for this job
            task_ids = self._tasks_by_job.pop(job_id, [])
            task_id_set = set(task_ids)
            for task_id in task_ids:
                self._tasks.pop(task_id, None)
            # Filter removed tasks from the priority queue in one pass
            self._task_queue = [entry for entry in self._task_queue if entry.task_id not in task_id_set]

            # Remove endpoints for the job
            self.remove_endpoints_for_job(job_id)

            # Remove the job itself
            self._jobs.pop(job_id, None)

            logger.info("Removed finished job %s (state=%s)", job_id, cluster_pb2.JobState.Name(job.state))
            return True

    # --- Task Management ---

    def get_task(self, task_id: JobName) -> ControllerTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_job_tasks(self, job_id: JobName) -> list[ControllerTask]:
        """Get all tasks for a job."""
        with self._lock:
            task_ids = self._tasks_by_job.get(job_id, [])
            return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def get_jobs_for_tasks(self, tasks: list[ControllerTask]) -> dict[JobName, ControllerJob]:
        """Return job lookup dict for all jobs referenced by the given tasks."""
        with self._lock:
            job_ids = {t.job_id for t in tasks}
            return {jid: self._jobs[jid] for jid in job_ids if jid in self._jobs}

    def get_tasks_for_worker(self, worker_id: WorkerId, limit: int = 50) -> list[ControllerTask]:
        """Return tasks that have been assigned to this worker, newest first."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return []
            matches = [self._tasks[tid] for tid in worker.task_history if tid in self._tasks]
            matches.sort(key=lambda t: t.started_at.epoch_ms() if t.started_at else 0, reverse=True)
            return matches[:limit]

    def peek_pending_tasks(self) -> list[ControllerTask]:
        """Return all schedulable tasks in priority order without removing them.

        A task is schedulable if it has no attempts yet (fresh task) or
        its current attempt is terminal and it should retry.
        """
        with self._lock:
            pending = []
            for entry in self._task_queue:
                task = self._tasks.get(entry.task_id)
                if task and task.can_be_scheduled():
                    pending.append(task)
            return pending

    def _finalize_job_state(
        self,
        job: ControllerJob,
        new_state: int,
        txn: TransactionLog,
    ) -> set[JobName]:
        """Apply a job state change and handle associated actions.

        Returns:
            Set of JobNames that were killed and had workers assigned. These need
            KILL RPCs sent to workers. Empty for SUCCEEDED state; populated for
            FAILED, KILLED, and UNSCHEDULABLE states.
        """
        job.state = new_state
        job.finished_at = Timestamp.now()

        killed_tasks: set[JobName] = set()
        if new_state == cluster_pb2.JOB_STATE_FAILED:
            job.error = self._get_first_task_error(job.job_id)
            killed_tasks = self._mark_remaining_tasks_killed(job.job_id, "Job exceeded max_task_failures", txn)
        elif new_state == cluster_pb2.JOB_STATE_SUCCEEDED:
            pass
        elif new_state == cluster_pb2.JOB_STATE_KILLED:
            job.error = self._get_first_task_error(job.job_id)
            killed_tasks = self._mark_remaining_tasks_killed(job.job_id, "Job was terminated.", txn)
        elif new_state == cluster_pb2.JOB_STATE_UNSCHEDULABLE:
            job.error = self._get_first_task_error(job.job_id)
            killed_tasks = self._mark_remaining_tasks_killed(job.job_id, "Job could not be scheduled.", txn)

        # Cascade termination to child jobs when parent reaches terminal state
        if new_state != cluster_pb2.JOB_STATE_SUCCEEDED:
            child_killed_tasks = self._cancel_child_jobs(job.job_id, txn)
            killed_tasks.update(child_killed_tasks)

        return killed_tasks

    def _cancel_child_jobs(self, parent_job_id: JobName, txn: TransactionLog) -> set[JobName]:
        """Recursively cancel all child jobs when a parent job terminates.

        Must be called with _lock held. Traverses the job hierarchy and cancels
        all non-finished descendant jobs.

        Returns:
            Set of task IDs that need KILL RPCs sent to workers.
        """
        killed_tasks: set[JobName] = set()

        # Find all direct children (lock already held by caller)
        children = [j for j in self._jobs.values() if j.job_id.parent == parent_job_id]

        for child in children:
            if child.is_finished():
                continue

            logger.info(
                "Cancelling child job %s due to parent %s termination",
                child.job_id,
                parent_job_id,
            )

            # Cancel this child job
            cancel_event = JobCancelledEvent(
                job_id=child.job_id,
                reason=f"Parent job {parent_job_id} terminated",
            )
            self._on_job_cancelled(txn, cancel_event)

            # Collect tasks that need kill RPCs
            for task_id in self._tasks_by_job.get(child.job_id, []):
                task = self._tasks.get(task_id)
                if task and task.worker_id:
                    killed_tasks.add(task_id)

        return killed_tasks

    def _get_first_task_error(self, job_id: JobName) -> str | None:
        """Get the first error message from failed/killed tasks in a job."""
        for task_id in self._tasks_by_job.get(job_id, []):
            task = self._tasks.get(task_id)
            if task and task.error:
                return task.error
        return None

    def _mark_remaining_tasks_killed(
        self,
        job_id: JobName,
        error: str,
        txn: TransactionLog,
    ) -> set[JobName]:
        """Mark all non-finished tasks in a job as killed (state-only).

        Updates job.task_state_counts incrementally for each killed task.
        The actual KILL RPCs to workers happen elsewhere; this method only
        updates internal state.

        Returns the set of killed JobNames that had workers assigned
        (these are the tasks that need KILL RPCs).
        """
        job = self._jobs.get(job_id)
        now = Timestamp.now()
        tasks_needing_kill_rpc: set[JobName] = set()
        tasks_to_remove_from_queue: set[JobName] = set()

        for task_id in self._tasks_by_job.get(job_id, []):
            task = self._tasks.get(task_id)
            if not task or task.is_finished():
                continue

            old_state = task.state
            had_worker = task.worker_id is not None

            task.state = cluster_pb2.TASK_STATE_KILLED
            task.finished_at = now
            task.error = error

            # Update job's task state counts
            if job:
                job.task_state_counts[old_state] -= 1
                job.task_state_counts[cluster_pb2.TASK_STATE_KILLED] += 1

            tasks_to_remove_from_queue.add(task_id)

            # Only track tasks with workers assigned for kill RPC
            if had_worker:
                tasks_needing_kill_rpc.add(task_id)

            if task.worker_id and job:
                worker = self._workers.get(task.worker_id)
                if worker:
                    worker.unassign_task(task_id, job.request.resources)
            self._remove_endpoints_for_task(task_id)

            txn.log("task_killed", task_id, old_state=old_state, error=error)

        # Filter the queue once after collecting all task IDs to remove (O(N) instead of O(N²))
        if tasks_to_remove_from_queue:
            self._task_queue = [entry for entry in self._task_queue if entry.task_id not in tasks_to_remove_from_queue]

        return tasks_needing_kill_rpc

    # --- Worker Management ---

    def add_worker(self, worker: ControllerWorker) -> None:
        """Add worker directly to state (test utility).

        For production use, create Event(WORKER_REGISTERED) instead.
        This method bypasses event logging and is intended for test setup only.
        """
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

    def snapshot_building_counts(self) -> dict[WorkerId, int]:
        """Atomically count tasks in BUILDING/ASSIGNED state per worker.

        Must be done under the lock because the scheduling thread would otherwise
        iterate worker.running_tasks while the heartbeat thread mutates it
        (RuntimeError: Set changed size during iteration).
        """
        with self._lock:
            counts: dict[WorkerId, int] = {}
            for worker in self._workers.values():
                count = 0
                for task_id in worker.running_tasks:
                    task = self._tasks.get(task_id)
                    if task and task.state in (
                        cluster_pb2.TASK_STATE_BUILDING,
                        cluster_pb2.TASK_STATE_ASSIGNED,
                    ):
                        count += 1
                if count > 0:
                    counts[worker.worker_id] = count
            return counts

    # =========================================================================
    # Heartbeat Dispatch API
    # =========================================================================

    def buffer_dispatch(self, worker_id: WorkerId, task_request: cluster_pb2.Worker.RunTaskRequest) -> None:
        """Buffer a task dispatch for the next heartbeat.

        Called by the scheduling thread after committing resources via TaskAssignedEvent.
        The dispatch will be delivered when begin_heartbeat() drains the buffer.
        """
        with self._lock:
            pd = self._pending_dispatch.setdefault(worker_id, PendingDispatch())
            pd.tasks_to_run.append(task_request)

    def buffer_kill(self, worker_id: WorkerId, task_id: str) -> None:
        """Buffer a task kill for the next heartbeat.

        Called when a task needs to be terminated on a worker. The kill will be
        delivered when begin_heartbeat() drains the buffer.
        """
        with self._lock:
            pd = self._pending_dispatch.setdefault(worker_id, PendingDispatch())
            pd.tasks_to_kill.append(task_id)

    def begin_heartbeat(self, worker_id: WorkerId) -> HeartbeatSnapshot | None:
        """Atomically snapshot worker state and drain dispatch buffers.

        Returns None if worker is no longer registered (removed while heartbeat pending).
        The snapshot is immutable and can be used for RPC without holding locks.
        """
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return None
            dispatch = self._pending_dispatch.pop(worker_id, PendingDispatch())
            running = []
            for tid in worker.running_tasks:
                task = self._tasks.get(tid)
                running.append(RunningTaskEntry(tid, task.current_attempt_id if task else 0))
            return HeartbeatSnapshot(
                worker_id=worker.worker_id,
                worker_address=worker.address,
                vm_address=worker.metadata.vm_address or "",
                running_tasks=running,
                tasks_to_run=dispatch.tasks_to_run,
                tasks_to_kill=dispatch.tasks_to_kill,
            )

    def complete_heartbeat(
        self,
        snapshot: HeartbeatSnapshot,
        response: cluster_pb2.HeartbeatResponse,
    ) -> None:
        """Process successful heartbeat response.

        Updates worker health state and processes task state changes from the response.
        """
        with self._lock:
            worker = self._workers.get(snapshot.worker_id)
            if not worker:
                return  # Worker removed while heartbeat in flight

            worker.last_heartbeat = Timestamp.now()
            worker.healthy = True
            worker.consecutive_failures = 0

            # Process running task state updates (e.g. ASSIGNED -> BUILDING -> RUNNING)
            for entry in response.running_tasks:
                if entry.state == cluster_pb2.TASK_STATE_UNSPECIFIED:
                    continue
                task_id = JobName.from_wire(entry.task_id)
                task = self._tasks.get(task_id)
                if task and not task.is_finished():
                    # Always persist log_directory as soon as the worker reports it,
                    # regardless of whether the state changed. This ensures we capture
                    # it even if the worker crashes before the next heartbeat cycle.
                    if entry.log_directory and entry.attempt_id < len(task.attempts):
                        if not task.attempts[entry.attempt_id].log_directory:
                            task.attempts[entry.attempt_id].log_directory = entry.log_directory
                    if task.state != entry.state:
                        # Ignore PENDING reported by the worker: the task thread starts in
                        # PENDING before transitioning to BUILDING, so the first heartbeat
                        # after assignment can carry a stale PENDING.  Accepting it would
                        # regress an ASSIGNED task back to PENDING and silently drop it
                        # from the building-count backpressure window.
                        if entry.state == cluster_pb2.TASK_STATE_PENDING:
                            continue
                        self._process_task_state_change(task_id, entry.state, entry.attempt_id)

            # Process completed tasks
            for entry in response.completed_tasks:
                task_id = JobName.from_wire(entry.task_id)
                task = self._tasks.get(task_id)
                if task and not task.is_finished():
                    self._process_task_state_change(
                        task_id,
                        entry.state,
                        entry.attempt_id,
                        error=entry.error or None,
                        exit_code=entry.exit_code,
                    )
                    # Store log_directory from worker
                    if task and entry.attempt_id < len(task.attempts) and entry.log_directory:
                        task.attempts[entry.attempt_id].log_directory = entry.log_directory

    def _process_task_state_change(
        self,
        task_id: JobName,
        new_state: int,
        attempt_id: int,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Internal helper to process a task state change within complete_heartbeat.

        Must be called with _lock held.
        """
        from iris.cluster.controller.events import TaskStateChangedEvent

        event = TaskStateChangedEvent(
            task_id=task_id,
            new_state=new_state,
            attempt_id=attempt_id,
            error=error,
            exit_code=exit_code,
        )
        txn = TransactionLog(event=event)
        self._on_task_state_changed(txn, event)
        self._transactions.append(txn)

    def fail_heartbeat(self, snapshot: HeartbeatSnapshot, error: str) -> None:
        """Handle heartbeat failure - requeue dispatches, track failures.

        Requeues any buffered dispatches so they can be retried on the next heartbeat.
        Increments the worker's consecutive failure count. If the threshold is exceeded,
        the worker will be marked as failed and pending dispatches are cleared (tasks
        will be requeued via WORKER_FAILED state transition).
        """
        with self._lock:
            worker = self._workers.get(snapshot.worker_id)
            if not worker:
                return  # Worker removed while heartbeat in flight

            # Record failure via event first - this may mark worker as failed
            event = WorkerHeartbeatFailedEvent(worker_id=snapshot.worker_id, error=error)
            txn = TransactionLog(event=event)
            self._on_worker_heartbeat_failed(txn, event)
            self._transactions.append(txn)

            # Only requeue dispatches if worker is still healthy.
            # If worker failed, tasks are already being requeued via WORKER_FAILED
            # state transition, and we don't want stale dispatches lingering.
            if worker.healthy:
                pd = self._pending_dispatch.setdefault(snapshot.worker_id, PendingDispatch())
                pd.tasks_to_run.extend(snapshot.tasks_to_run)
                pd.tasks_to_kill.extend(snapshot.tasks_to_kill)
            else:
                # Worker failed - clear any pending dispatches
                self._pending_dispatch.pop(snapshot.worker_id, None)

    def load_workers_from_config(self, configs: list[WorkerConfig]) -> None:
        """Load workers from static configuration."""
        now = Timestamp.now()
        for cfg in configs:
            worker = ControllerWorker(
                worker_id=WorkerId(cfg.worker_id),
                address=cfg.address,
                metadata=cfg.metadata,
                last_heartbeat=now,
            )
            self.add_worker(worker)

    # --- Endpoint Management ---

    def add_endpoint(self, endpoint: ControllerEndpoint, task_id: JobName | None = None) -> None:
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

    def _visible_endpoints(self, predicate: Callable[[ControllerEndpoint], bool]) -> list[ControllerEndpoint]:
        """Return endpoints matching predicate whose jobs are in non-terminal states."""
        results = []
        for ep in self._endpoints.values():
            if not predicate(ep):
                continue
            job = self._jobs.get(ep.job_id)
            if job and not job.is_finished():
                results.append(ep)
        return results

    def lookup_endpoints(self, name: str) -> list[ControllerEndpoint]:
        """Find endpoints by exact name match for non-terminal jobs."""
        with self._lock:
            return self._visible_endpoints(lambda ep: ep.name == name)

    def list_endpoints_by_prefix(self, prefix: str) -> list[ControllerEndpoint]:
        """List endpoints matching a name prefix for non-terminal jobs."""
        with self._lock:
            return self._visible_endpoints(lambda ep: ep.name.startswith(prefix))

    def list_all_endpoints(self) -> list[ControllerEndpoint]:
        """Return all registered endpoints."""
        with self._lock:
            return list(self._endpoints.values())

    def _remove_endpoints_for_task(self, task_id: JobName) -> list[ControllerEndpoint]:
        """Remove all endpoints associated with a task."""
        endpoint_ids = list(self._endpoints_by_task.get(task_id, []))
        removed = []
        for eid in endpoint_ids:
            endpoint = self._endpoints.pop(eid, None)
            if endpoint:
                removed.append(endpoint)
        self._endpoints_by_task.pop(task_id, None)
        return removed

    def remove_endpoints_for_job(self, job_id: JobName) -> list[ControllerEndpoint]:
        """Remove all endpoints for a job by removing endpoints for all its tasks."""
        with self._lock:
            all_removed = []
            for task_id in self._tasks_by_job.get(job_id, []):
                removed = self._remove_endpoints_for_task(task_id)
                all_removed.extend(removed)
            return all_removed
