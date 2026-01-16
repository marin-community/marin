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

from iris.cluster.controller.job import Job, TransitionResult
from iris.cluster.types import JobId, WorkerId
from iris.rpc import cluster_pb2

# Backwards compatibility alias
ControllerJob = Job


@dataclass
class WorkerConfig:
    """Static worker configuration for v0.

    Args:
        worker_id: Unique worker identifier
        address: Worker RPC address (host:port)
        resources: Worker's available resources
    """

    worker_id: str
    address: str
    resources: cluster_pb2.ResourceSpec


@dataclass
class ControllerWorker:
    """Controller's view of a worker.

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
    """Thread-safe controller state managing jobs, workers, endpoints, and the job queue."""

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
        with self._lock:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            if job.gang_id:
                self._gangs.setdefault(job.gang_id, set()).add(job.job_id)

    def get_job(self, job_id: JobId) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def pop_next_pending(self) -> Job | None:
        """Pop next PENDING job from the queue, skipping jobs that have transitioned to other states."""
        with self._lock:
            while self._queue:
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job and job.state == cluster_pb2.JOB_STATE_PENDING:
                    return job
            return None

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

    def list_all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def list_all_workers(self) -> list[ControllerWorker]:
        with self._lock:
            return list(self._workers.values())

    def get_gang_jobs(self, gang_id: str) -> list[Job]:
        with self._lock:
            job_ids = self._gangs.get(gang_id, set())
            return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    def get_children(self, job_id: JobId) -> list[Job]:
        with self._lock:
            return [job for job in self._jobs.values() if job.parent_job_id == job_id]

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

    def peek_pending_jobs(self) -> list[Job]:
        """Return all PENDING jobs in queue order without removing them."""
        with self._lock:
            pending = []
            for job_id in self._queue:
                job = self._jobs.get(job_id)
                if job and job.state == cluster_pb2.JOB_STATE_PENDING:
                    pending.append(job)
            return pending

    def _add_to_queue(self, job_id: JobId) -> None:
        """Internal: add job to queue if not already present. Caller must hold lock."""
        if job_id not in self._queue:
            self._queue.append(job_id)

    def add_endpoint(self, endpoint: ControllerEndpoint) -> None:
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            self._endpoints_by_job.setdefault(endpoint.job_id, set()).add(endpoint.endpoint_id)

    def remove_endpoint(self, endpoint_id: str) -> ControllerEndpoint | None:
        with self._lock:
            endpoint = self._endpoints.pop(endpoint_id, None)
            if endpoint:
                job_endpoints = self._endpoints_by_job.get(endpoint.job_id)
                if job_endpoints:
                    job_endpoints.discard(endpoint_id)
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

    def remove_endpoints_for_job(self, job_id: JobId) -> list[ControllerEndpoint]:
        with self._lock:
            return self._remove_endpoints_for_job(job_id)

    def _remove_endpoints_for_job(self, job_id: JobId) -> list[ControllerEndpoint]:
        endpoint_ids = list(self._endpoints_by_job.get(job_id, []))
        removed = []
        for eid in endpoint_ids:
            endpoint = self._endpoints.pop(eid, None)
            if endpoint:
                job_endpoints = self._endpoints_by_job.get(endpoint.job_id)
                if job_endpoints:
                    job_endpoints.discard(eid)
                removed.append(endpoint)
        self._endpoints_by_job.pop(job_id, None)
        return removed

    # --- Worker Mutation Methods ---

    def assign_job_to_worker(self, worker_id: WorkerId, job_id: JobId) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.running_jobs.add(job_id)
            self._queue = deque(jid for jid in self._queue if jid != job_id)
            return True

    def unassign_job_from_worker(self, worker_id: WorkerId, job_id: JobId) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.running_jobs.discard(job_id)
            return True

    def rollback_assignment(self, worker_id: WorkerId, job: Job) -> None:
        """Rollback a failed assignment: unassign from worker and re-queue."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.running_jobs.discard(job.job_id)
            self._add_to_queue(job.job_id)

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
        resources: cluster_pb2.ResourceSpec | None = None,
        metadata: cluster_pb2.WorkerMetadata | None = None,
    ) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.last_heartbeat_ms = now_ms
            worker.healthy = True
            if resources is not None:
                worker.resources = resources
            if metadata is not None:
                worker.metadata = metadata
            return True

    # --- Job Finalization ---

    def transition_job(
        self,
        job_id: JobId,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> tuple[TransitionResult, list[ControllerEndpoint]]:
        """Transition a job and handle all side effects automatically.

        Side effects based on result:
        - SHOULD_RETRY: Re-queues job, unassigns from worker
        - Terminal state: Removes from queue, removes endpoints, unassigns from worker
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return TransitionResult.COMPLETE, []

            # Capture worker_id before transition (reset_for_retry clears it)
            worker_id = job.worker_id

            result = job.transition(
                new_state,
                now_ms,
                is_worker_failure=is_worker_failure,
                error=error,
                exit_code=exit_code,
            )

            removed_endpoints: list[ControllerEndpoint] = []
            if result == TransitionResult.SHOULD_RETRY:
                if job.job_id not in self._queue:
                    self._queue.append(job.job_id)
                if worker_id:
                    worker = self._workers.get(worker_id)
                    if worker:
                        worker.running_jobs.discard(job_id)
            elif job.is_finished():
                self._queue = deque(jid for jid in self._queue if jid != job_id)
                removed_endpoints = self._remove_endpoints_for_job(job_id)
                if worker_id:
                    worker = self._workers.get(worker_id)
                    if worker:
                        worker.running_jobs.discard(job_id)

            return result, removed_endpoints

    # --- Resource and Scheduling Methods ---

    def get_committed_resources(self, worker: ControllerWorker) -> tuple[int, int, int]:
        """Compute resources committed to running jobs on this worker.

        Returns:
            (cpu, memory_bytes, gpu_count) tuple of committed resources
        """
        from iris.cluster.controller.resources import get_gpu_count

        with self._lock:
            cpu = 0
            memory = 0
            gpu = 0

            for job_id in worker.running_jobs:
                job = self._jobs.get(job_id)
                if job:
                    resources = job.request.resources
                    cpu += resources.cpu
                    memory += resources.memory_bytes
                    gpu += get_gpu_count(resources.device)

            return cpu, memory, gpu

    def load_workers_from_config(self, configs: list[WorkerConfig]) -> None:
        """Load workers from static configuration."""
        now_ms = int(time.time() * 1000)
        for cfg in configs:
            worker = ControllerWorker(
                worker_id=WorkerId(cfg.worker_id),
                address=cfg.address,
                resources=cfg.resources,
                last_heartbeat_ms=now_ms,
            )
            self.add_worker(worker)
