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

"""Controller core data structures.

This module provides the in-memory state management for the controller, including:
- Job: Job with self-contained state transitions (imported from job.py)
- ControllerJob: Alias for Job (for backwards compatibility)
- ControllerWorker: Controller's view of a worker with health and capacity
- ControllerEndpoint: An endpoint registered with the controller for service discovery
- ActionLogEntry: Record of a controller action for the dashboard
- ControllerState: Thread-safe state container for jobs, workers, endpoints, and the queue

All state mutations are protected by a reentrant lock (RLock) to support
concurrent access from the scheduler, heartbeat monitor, and RPC handlers.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock

from fluster.cluster.controller.job import Job
from fluster.cluster.types import JobId, WorkerId
from fluster.rpc import cluster_pb2

# Backwards compatibility alias
ControllerJob = Job


@dataclass
class ControllerWorker:
    """Controller's view of a worker.

    Tracks worker capabilities, health status, and current job assignments.
    Workers send periodic heartbeats to the controller, which uses this to
    detect worker failures. The scheduler uses this to find available capacity.

    Args:
        worker_id: Unique worker identifier
        address: Worker RPC address (host:port)
        resources: Worker's available resources
        metadata: Worker environment metadata (TPU vars, GPU info, etc.)
        healthy: Whether worker is currently healthy
        consecutive_failures: Number of consecutive heartbeat failures
        last_heartbeat_ms: Timestamp of last successful heartbeat
        running_jobs: Set of job IDs currently running on this worker
    """

    worker_id: WorkerId
    address: str
    resources: cluster_pb2.ResourceSpec
    metadata: cluster_pb2.WorkerMetadata | None = None

    # Health tracking
    healthy: bool = True
    consecutive_failures: int = 0
    last_heartbeat_ms: int = 0

    # Current assignments
    running_jobs: set[JobId] = field(default_factory=set)


@dataclass
class ControllerEndpoint:
    """An endpoint registered with the controller.

    Endpoints are associated with jobs and used for service discovery.
    When a job transitions to a terminal state, all its endpoints are
    automatically removed.

    Names are prefixed with the root job ID for namespace isolation:
    e.g., "abc123/my-actor" where "abc123" is the root job ID.

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

    Actions are logged when significant events occur (job submitted, job started,
    worker registered, etc.) to provide visibility into controller activity.

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
    """Thread-safe controller state.

    Manages in-memory state for jobs, workers, endpoints, job queue, and gang tracking.
    All mutations go through methods that acquire the lock to ensure consistency
    during concurrent access from multiple threads (scheduler, heartbeat monitor,
    RPC handlers).

    The job queue is FIFO by default, with pop_next_pending() automatically
    skipping jobs that are no longer in PENDING state.

    Endpoints are associated with jobs and automatically removed when jobs
    transition to terminal states. Lookup and list operations filter to only
    return endpoints for jobs in RUNNING state.
    """

    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[JobId, Job] = {}
        self._workers: dict[WorkerId, ControllerWorker] = {}
        self._queue: deque[JobId] = deque()  # FIFO queue of job IDs
        self._gangs: dict[str, set[JobId]] = {}  # gang_id -> job_ids
        self._actions: deque[ActionLogEntry] = deque(maxlen=100)  # Recent actions log
        self._endpoints: dict[str, ControllerEndpoint] = {}  # endpoint_id -> endpoint
        self._endpoints_by_job: dict[JobId, set[str]] = {}  # job_id -> endpoint_ids

    def add_job(self, job: Job) -> None:
        """Add a job to the controller state and queue.

        The job is added to the jobs dict, appended to the FIFO queue, and
        registered with its gang (if it has one).

        Args:
            job: Job to add
        """
        with self._lock:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            if job.gang_id:
                self._gangs.setdefault(job.gang_id, set()).add(job.job_id)

    def get_job(self, job_id: JobId) -> Job | None:
        """Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job if found, None otherwise
        """
        with self._lock:
            return self._jobs.get(job_id)

    def pop_next_pending(self) -> Job | None:
        """Pop next PENDING job from the queue.

        Iterates through the queue until finding a job in PENDING state,
        skipping jobs that have transitioned to other states.

        Returns:
            Next pending job, or None if queue is empty or no pending jobs
        """
        with self._lock:
            while self._queue:
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job and job.state == cluster_pb2.JOB_STATE_PENDING:
                    return job
            return None

    def add_worker(self, worker: ControllerWorker) -> None:
        """Add or update a worker in the registry.

        Args:
            worker: Worker to add
        """
        with self._lock:
            self._workers[worker.worker_id] = worker

    def get_worker(self, worker_id: WorkerId) -> ControllerWorker | None:
        """Get a worker by ID.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker if found, None otherwise
        """
        with self._lock:
            return self._workers.get(worker_id)

    def remove_worker(self, worker_id: WorkerId) -> ControllerWorker | None:
        """Remove a worker from the registry.

        Used when a worker is permanently gone (e.g., after exceeding
        heartbeat failure threshold).

        Args:
            worker_id: Worker identifier

        Returns:
            Removed worker if found, None otherwise
        """
        with self._lock:
            return self._workers.pop(worker_id, None)

    def get_available_workers(self) -> list[ControllerWorker]:
        """Return healthy workers with capacity.

        For v0, simply returns all healthy workers. Future versions will
        filter by available resources and capacity constraints.

        Returns:
            List of healthy workers
        """
        with self._lock:
            return [w for w in self._workers.values() if w.healthy]

    def list_all_jobs(self) -> list[Job]:
        """Get all jobs in the controller.

        Returns:
            List of all jobs (snapshot under lock)
        """
        with self._lock:
            return list(self._jobs.values())

    def list_all_workers(self) -> list[ControllerWorker]:
        """Get all workers in the controller.

        Returns:
            List of all workers (snapshot under lock)
        """
        with self._lock:
            return list(self._workers.values())

    def get_gang_jobs(self, gang_id: str) -> list[Job]:
        """Get all jobs in a gang.

        Args:
            gang_id: Gang identifier

        Returns:
            List of jobs in the gang (may be empty)
        """
        with self._lock:
            job_ids = self._gangs.get(gang_id, set())
            return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    def get_children(self, job_id: JobId) -> list[Job]:
        """Get all direct child jobs of a parent job.

        Args:
            job_id: Parent job identifier

        Returns:
            List of child jobs (may be empty)
        """
        with self._lock:
            return [job for job in self._jobs.values() if job.parent_job_id == job_id]

    def log_action(
        self,
        action: str,
        job_id: JobId | None = None,
        worker_id: WorkerId | None = None,
        details: str = "",
    ) -> None:
        """Record an action in the log.

        Actions are stored in a bounded deque (last 100 entries) for display
        on the dashboard.

        Args:
            action: Action type (e.g., "job_submitted", "job_started")
            job_id: Associated job ID, if any
            worker_id: Associated worker ID, if any
            details: Additional human-readable details
        """
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
        """Get most recent actions.

        Args:
            limit: Maximum number of actions to return

        Returns:
            List of recent actions, most recent last
        """
        with self._lock:
            actions = list(self._actions)
            return actions[-limit:] if limit < len(actions) else actions

    def peek_pending_jobs(self) -> list[Job]:
        """Return all PENDING jobs in queue order without removing them.

        Used by the scheduler to iterate through the queue and find schedulable
        jobs. Unlike pop_next_pending(), this returns all pending jobs so the
        scheduler can skip jobs that don't fit and continue to the next.

        Returns:
            List of pending jobs in FIFO order
        """
        with self._lock:
            pending = []
            for job_id in self._queue:
                job = self._jobs.get(job_id)
                if job and job.state == cluster_pb2.JOB_STATE_PENDING:
                    pending.append(job)
            return pending

    def remove_from_queue(self, job_id: JobId) -> None:
        """Remove a specific job from the queue.
        Args:
            job_id: Job ID to remove from the queue
        """
        with self._lock:
            self._queue = deque(jid for jid in self._queue if jid != job_id)

    def add_to_queue(self, job: Job) -> None:
        """Add a job back to the queue for retry.

        Used when dispatch fails and the job needs to be rescheduled.
        The job must already exist in the jobs dict.

        Args:
            job: Job to add back to the queue
        """
        with self._lock:
            if job.job_id not in self._queue:
                self._queue.append(job.job_id)

    def add_endpoint(self, endpoint: ControllerEndpoint) -> None:
        """Add an endpoint to the controller registry.

        Endpoints are tracked both by ID and by job. When a job terminates,
        all its endpoints can be quickly removed.

        Args:
            endpoint: Endpoint to register
        """
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            self._endpoints_by_job.setdefault(endpoint.job_id, set()).add(endpoint.endpoint_id)

    def remove_endpoint(self, endpoint_id: str) -> ControllerEndpoint | None:
        """Remove an endpoint from the registry.

        Args:
            endpoint_id: Endpoint ID to remove

        Returns:
            Removed endpoint if found, None otherwise
        """
        with self._lock:
            endpoint = self._endpoints.pop(endpoint_id, None)
            if endpoint:
                job_endpoints = self._endpoints_by_job.get(endpoint.job_id)
                if job_endpoints:
                    job_endpoints.discard(endpoint_id)
            return endpoint

    def lookup_endpoints(self, name: str) -> list[ControllerEndpoint]:
        """Find endpoints by exact name match.

        Only returns endpoints for jobs in RUNNING state. Endpoints for
        jobs that have terminated or are not yet running are filtered out.

        Names are expected to be fully prefixed (e.g., "abc123/my-actor").

        Args:
            name: Full prefixed service name to look up

        Returns:
            List of matching endpoints (may be empty)
        """
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
        """List endpoints matching a name prefix.

        Only returns endpoints for jobs in RUNNING state.

        Prefix is expected to include the namespace (e.g., "abc123/" or "abc123/my-actor").

        Args:
            prefix: Service name prefix to match (includes namespace prefix)

        Returns:
            List of matching endpoints (may be empty)
        """
        with self._lock:
            results = []
            for ep in self._endpoints.values():
                if not ep.name.startswith(prefix):
                    continue
                job = self._jobs.get(ep.job_id)
                if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                    results.append(ep)
            return results

    def remove_endpoints_for_job(self, job_id: JobId) -> list[ControllerEndpoint]:
        """Remove all endpoints for a job.

        Called when a job transitions to a terminal state to clean up
        all service discovery entries.

        Args:
            job_id: Job ID whose endpoints should be removed

        Returns:
            List of removed endpoints
        """
        with self._lock:
            endpoint_ids = list(self._endpoints_by_job.get(job_id, []))
            removed = []
            for eid in endpoint_ids:
                ep = self.remove_endpoint(eid)
                if ep:
                    removed.append(ep)
            # Clean up the job mapping
            self._endpoints_by_job.pop(job_id, None)
            return removed
