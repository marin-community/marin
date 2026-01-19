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

"""Controller core data structures and thread-safe state container."""

import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock

from iris.cluster.controller.job import Job, expand_job_to_tasks
from iris.cluster.controller.task import Task, TaskTransitionResult
from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms


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


@dataclass
class ActionLogEntry:
    """Record of a controller action for the dashboard.

    Args:
        timestamp_ms: Unix timestamp in milliseconds when action occurred
        action: Action type (e.g., "job_submitted", "job_started", "worker_failed")
        job_id: Associated job ID, if any
        worker_id: Associated worker ID, if any
        details: Additional human-readable details
    """

    timestamp_ms: int
    action: str
    job_id: JobId | None = None
    worker_id: WorkerId | None = None
    details: str = ""


class ControllerState:
    """Thread-safe controller state managing jobs, tasks, workers, endpoints, and queues."""

    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[JobId, Job] = {}
        self._tasks: dict[TaskId, Task] = {}
        self._tasks_by_job: dict[JobId, list[TaskId]] = {}
        self._workers: dict[WorkerId, ControllerWorker] = {}
        self._task_queue: deque[TaskId] = deque()  # FIFO queue of task IDs
        self._gangs: dict[str, set[JobId]] = {}  # gang_id -> job_ids
        self._actions: deque[ActionLogEntry] = deque(maxlen=100)  # Recent actions log
        self._endpoints: dict[str, ControllerEndpoint] = {}  # endpoint_id -> endpoint
        self._endpoints_by_task: dict[TaskId, set[str]] = {}  # task_id -> endpoint_ids

    # --- Job Management ---

    def add_job(self, job: Job, tasks: list[Task] | None = None) -> list[Task]:
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
                tasks = expand_job_to_tasks(job, now_ms())

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

    def get_job(self, job_id: JobId) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get_gang_jobs(self, gang_id: str) -> list[Job]:
        with self._lock:
            job_ids = self._gangs.get(gang_id, set())
            return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    def get_children(self, job_id: JobId) -> list[Job]:
        with self._lock:
            return [job for job in self._jobs.values() if job.parent_job_id == job_id]

    # --- Task Management ---

    def get_task(self, task_id: TaskId) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_job_tasks(self, job_id: JobId) -> list[Task]:
        """Get all tasks for a job."""
        with self._lock:
            task_ids = self._tasks_by_job.get(job_id, [])
            return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def peek_pending_tasks(self) -> list[Task]:
        """Return all PENDING tasks in queue order without removing them."""
        with self._lock:
            pending = []
            for task_id in self._task_queue:
                task = self._tasks.get(task_id)
                if task and task.state == cluster_pb2.TASK_STATE_PENDING:
                    pending.append(task)
            return pending

    def assign_task_to_worker(self, worker_id: WorkerId, task_id: TaskId) -> bool:
        """Assign a task to a worker and remove from queue."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.running_tasks.add(task_id)
            self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
            return True

    def rollback_task_assignment(self, worker_id: WorkerId, task: Task) -> None:
        """Rollback a failed assignment: unassign from worker and re-queue."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.running_tasks.discard(task.task_id)
            if task.task_id not in self._task_queue:
                self._task_queue.append(task.task_id)

    def mark_task_dispatched(self, task: Task, worker_id: WorkerId, now_ms: int) -> int | None:
        """Mark a task as dispatched and update job state counts.

        Returns:
            New job state if changed, None otherwise
        """
        with self._lock:
            job = self._jobs.get(task.job_id)
            if not job:
                task.mark_dispatched(worker_id, now_ms)
                return None

            old_state = task.state
            task.mark_dispatched(worker_id, now_ms)
            new_job_state = job.on_task_transition(old_state, task.state, now_ms)
            if new_job_state is not None:
                job.state = new_job_state
            return new_job_state

    def revert_task_dispatch(self, task: Task) -> None:
        """Revert a failed dispatch and update job state counts."""
        with self._lock:
            job = self._jobs.get(task.job_id)
            old_state = task.state
            task.revert_dispatch()
            if job:
                job.task_state_counts[old_state] -= 1
                job.task_state_counts[task.state] += 1

    def transition_task(
        self,
        task_id: TaskId,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> tuple[TaskTransitionResult, list[ControllerEndpoint]]:
        """Transition a task and handle all side effects automatically.

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

            result = task.transition(
                new_state,
                now_ms,
                is_worker_failure=is_worker_failure,
                error=error,
                exit_code=exit_code,
            )

            # Update job's task state counts incrementally
            new_job_state = job.on_task_transition(old_state, task.state, now_ms)

            removed_endpoints: list[ControllerEndpoint] = []
            if result == TaskTransitionResult.SHOULD_RETRY:
                if task.task_id not in self._task_queue:
                    self._task_queue.append(task.task_id)
                if worker_id:
                    worker = self._workers.get(worker_id)
                    if worker:
                        worker.running_tasks.discard(task_id)
            elif task.is_finished():
                self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
                removed_endpoints = self._remove_endpoints_for_task(task_id)
                if worker_id:
                    worker = self._workers.get(worker_id)
                    if worker:
                        worker.running_tasks.discard(task_id)

            # Apply job state change if needed
            if new_job_state is not None:
                job.state = new_job_state
                if new_job_state == cluster_pb2.JOB_STATE_FAILED:
                    job.finished_at_ms = now_ms
                    job.error = self._get_first_task_error(task.job_id)
                    failed_count = job.task_state_counts[cluster_pb2.TASK_STATE_FAILED]
                    self.log_action(
                        "failure_domain_triggered",
                        job_id=task.job_id,
                        details=f"failed={failed_count} max_allowed={job.request.max_task_failures}",
                    )
                    self._kill_remaining_tasks(task.job_id, now_ms, "Job exceeded max_task_failures")
                elif new_job_state in (
                    cluster_pb2.JOB_STATE_SUCCEEDED,
                    cluster_pb2.JOB_STATE_KILLED,
                    cluster_pb2.JOB_STATE_UNSCHEDULABLE,
                ):
                    job.finished_at_ms = now_ms
                    if job.error is None:
                        job.error = self._get_first_task_error(task.job_id)

            return result, removed_endpoints

    def _get_first_task_error(self, job_id: JobId) -> str | None:
        """Get the first error message from failed/killed tasks in a job."""
        for task_id in self._tasks_by_job.get(job_id, []):
            task = self._tasks.get(task_id)
            if task and task.error:
                return task.error
        return None

    def _kill_remaining_tasks(self, job_id: JobId, now_ms: int, error: str) -> None:
        """Kill all non-finished tasks in a job (failure domain).

        Updates job.task_state_counts incrementally for each killed task.
        """
        job = self._jobs.get(job_id)
        killed_count = 0
        for task_id in self._tasks_by_job.get(job_id, []):
            task = self._tasks.get(task_id)
            if not task or task.is_finished():
                continue

            old_state = task.state
            task.state = cluster_pb2.TASK_STATE_KILLED
            task.finished_at_ms = now_ms
            task.error = error
            killed_count += 1

            # Update job's task state counts
            if job:
                job.task_state_counts[old_state] -= 1
                job.task_state_counts[cluster_pb2.TASK_STATE_KILLED] += 1

            self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
            if task.worker_id:
                worker = self._workers.get(task.worker_id)
                if worker:
                    worker.running_tasks.discard(task_id)
            self._remove_endpoints_for_task(task_id)

        if killed_count > 0:
            self.log_action(
                "tasks_killed",
                job_id=job_id,
                details=f"killed={killed_count} reason={error}",
            )

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
        now_ms = int(time.time() * 1000)
        for cfg in configs:
            worker = ControllerWorker(
                worker_id=WorkerId(cfg.worker_id),
                address=cfg.address,
                metadata=cfg.metadata,
                last_heartbeat_ms=now_ms,
            )
            self.add_worker(worker)

    # --- Action Log ---

    def log_action(
        self,
        action: str,
        job_id: JobId | None = None,
        worker_id: WorkerId | None = None,
        details: str = "",
    ) -> None:
        entry = ActionLogEntry(
            timestamp_ms=int(time.time() * 1000),
            action=action,
            job_id=job_id,
            worker_id=worker_id,
            details=details,
        )
        with self._lock:
            self._actions.append(entry)

    def get_recent_actions(self, limit: int = 50) -> list[ActionLogEntry]:
        with self._lock:
            actions = list(self._actions)
            return actions[-limit:] if limit < len(actions) else actions

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

    # --- Resource Tracking ---

    def get_committed_resources(self, worker: ControllerWorker) -> tuple[int, int, int]:
        """Compute resources committed to running tasks on this worker.

        Returns:
            (cpu, memory_bytes, gpu_count) tuple of committed resources
        """
        from iris.cluster.controller.scheduler import get_gpu_count

        with self._lock:
            cpu = 0
            memory = 0
            gpu = 0

            for task_id in worker.running_tasks:
                task = self._tasks.get(task_id)
                if task:
                    job = self._jobs.get(task.job_id)
                    if job:
                        resources = job.request.resources
                        cpu += resources.cpu
                        memory += resources.memory_bytes
                        gpu += get_gpu_count(resources.device)

            return cpu, memory, gpu

    def remove_endpoints_for_job(self, job_id: JobId) -> list[ControllerEndpoint]:
        """Remove all endpoints for a job by removing endpoints for all its tasks."""
        with self._lock:
            all_removed = []
            for task_id in self._tasks_by_job.get(job_id, []):
                removed = self._remove_endpoints_for_task(task_id)
                all_removed.extend(removed)
            return all_removed
