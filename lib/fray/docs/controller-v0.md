# Controller V0

We're working on the controller from `fray-zero.md`. This controller manages a
set of VMs, starts Fluster workers on them, and manages job dispatch and
monitoring of those workers. We are going to start implementation of this work.

We'll start with the server side before moving over to the user-facing client.

The server RPC protocol is defined in `cluster.proto`. It accepts new jobs,
registers workers, and allows users to monitor job status or terminate jobs as
needed. The controller based on a queueing system, where jobs are dispatched via
a priority queue. The controller has a notion of "users" and users can have
different numbers of credits.

### Scheduling

For our first implementation, we'll use a simple FIFO queueing system. The
controller will check if new jobs can be scheduled on the following basis:

* On a one second timer
* Whenever a reservation is terminated
* Whenever a new worker registers

The job scheduler runs in a separate thread and is woken in response to the above events.

### Worker Registration

The v0 cluster will accept a list of workers to use at startup. Future
iterations will allow workers to register themselves automatically via RPC.

Workers have a set of properties that define their capabilities, e.g. the
number of TPUs, amount of RAM, etc.

### Heartbeats and Health Checks

The controller will periodically e.g. 1/second check the health of all workers.

* The heartbeat will contain a "since" timestamp, and the worker will respond with
a list of all jobs currently running on that worker or which have been running
on that worker since the "since" timestamp.

* If the worker fails to resspond to N consecutive heartbeats, the controller will mark the worker as unhealthy and remove it from the list of available workers.
Jobs on that worker will be terminated with JOB_STATE_WORKER_FAILED status, and retried if eligible.

### Job Failure and Retries

Job failures come in 2 types: external (worker failures) and internal (job
failures). A job may specify how many of either type of failure it should
tolerate.

Jobs are retried at the cluster level. If a job fails for an internal reason, it
may be re-scheduled onto the same worker or a different worker.

### Gang Scheduling

TPU jobs are "gang-scheduled" onto a set of linked workers. If any of the
workers or jobs fails, _all_ jobs in the gang are terminated by the
controller.The job gang will be retried if eligible.

### Dashboard

The controller provides a web UI dashboard, which shows:

* Recent actions log, e.g. job started, job terminated, etc.
* Job queue showing all jobs in priority order
* List of users, with links to their jobs and credits
* List of reservations, with links to their jobs and available resources
* List of workers, with overview of the worker and links to the worker status and recent jobs on that worker

## Future Work

This work is deferred to a future iteration:

### Reservations

Jobs are run inside of "reservations", which define the maximum number of
resources e.g. RAM, TPUs available for the job. The reservation ID, as with all
Fray information, is communicated to a job via the FRAY_RESERVATION_ID
environment variable.

Reservations are tied to their initial job, and are cleaned up when that job completes.
Jobs are tied to their parent, and are cleaned up when that job completes.
Jobs run inside of a _namespace_, which is communicated to a job via the FRAY_NAMESPACE
environment variable.

### User Credits

Users have a set of credits that determine the maximum number of resources they
can use.  Users credits are used automatically to determine the priority of
jobs. As a user runs jobs, it depletes their available credits, with future jobs
running at a lower priority.

---

## Implementation Plan

This section defines a tight, incrementally testable implementation path. Each
stage builds on the previous and has explicit test checkpoints.

**Note on types.py**: The current `fluster/cluster/types.py` contains Python
dataclasses that may duplicate proto definitions. As we implement controller
features, prefer proto messages for wire-format types (`ResourceSpec`,
`JobStatus`, etc). Use Python dataclasses only for controller-internal state
that doesn't cross RPC boundaries (like `ControllerJob`, `ControllerWorker`).
Clean up redundant types as you go.

### Stage 1: Proto Updates for Controller-Worker Communication

**Goal**: Define the wire protocol for heartbeats, worker registration, and job state sync.

Add to `cluster.proto`:

```protobuf
// Worker registration and heartbeat

message WorkerInfo {
  string worker_id = 1;
  string address = 2;              // host:port for WorkerService
  ResourceSpec resources = 3;      // Worker capabilities
  int64 registered_at_ms = 4;
}

message RegisterWorkerRequest {
  string worker_id = 1;
  string address = 2;
  ResourceSpec resources = 3;
}

message RegisterWorkerResponse {
  bool accepted = 1;
  string controller_address = 2;   // For callbacks
}

message HeartbeatRequest {
  string worker_id = 1;
  int64 since_ms = 2;              // Return jobs modified since this timestamp
}

message HeartbeatResponse {
  repeated JobStatus jobs = 1;     // Jobs running/completed since since_ms
  int64 timestamp_ms = 2;
}

message WorkerHealthStatus {
  string worker_id = 1;
  bool healthy = 2;
  int32 consecutive_failures = 3;
  int64 last_heartbeat_ms = 4;
  repeated string running_job_ids = 5;
}

// Add JOB_STATE_WORKER_FAILED to JobState enum
// JOB_STATE_WORKER_FAILED = 7;  // Worker died, job may be retried

// Controller service additions
service ControllerService {
  // ... existing methods ...
  rpc RegisterWorker(RegisterWorkerRequest) returns (RegisterWorkerResponse);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
  rpc ListWorkers(Empty) returns (ListWorkersResponse);
}

message ListWorkersResponse {
  repeated WorkerHealthStatus workers = 1;
}
```

**Deliverable**: Updated proto, regenerated Python stubs.

**Test**: Proto compiles, types import correctly.

---

### Stage 2: Controller Core Data Structures

**Goal**: In-memory state for jobs, workers, and the queue.

```python
# lib/fluster/src/fluster/cluster/controller/state.py

from dataclasses import dataclass, field
from collections import deque
from threading import RLock
from typing import NewType

from fluster import cluster_pb2
from fluster.cluster.types import JobId, WorkerId

@dataclass
class ControllerJob:
    """Controller's view of a job."""
    job_id: JobId
    request: cluster_pb2.LaunchJobRequest
    state: int = cluster_pb2.JOB_STATE_PENDING
    worker_id: WorkerId | None = None

    # Retry tracking
    failure_count: int = 0
    preemption_count: int = 0
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Gang scheduling
    gang_id: str | None = None

    # Timestamps
    submitted_at_ms: int = 0
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    error: str | None = None
    exit_code: int | None = None


@dataclass
class ControllerWorker:
    """Controller's view of a worker."""
    worker_id: WorkerId
    address: str
    resources: cluster_pb2.ResourceSpec

    # Health tracking
    healthy: bool = True
    consecutive_failures: int = 0
    last_heartbeat_ms: int = 0

    # Current assignments
    running_jobs: set[JobId] = field(default_factory=set)


class ControllerState:
    """Thread-safe controller state.

    All mutations go through methods that acquire the lock.
    """

    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[JobId, ControllerJob] = {}
        self._workers: dict[WorkerId, ControllerWorker] = {}
        self._queue: deque[JobId] = deque()  # FIFO queue of PENDING jobs
        self._gangs: dict[str, set[JobId]] = {}  # gang_id -> job_ids

    def add_job(self, job: ControllerJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            if job.gang_id:
                self._gangs.setdefault(job.gang_id, set()).add(job.job_id)

    def get_job(self, job_id: JobId) -> ControllerJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def pop_next_pending(self) -> ControllerJob | None:
        """Pop next job from queue if available."""
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

    def get_available_workers(self) -> list[ControllerWorker]:
        """Return healthy workers with capacity."""
        with self._lock:
            return [w for w in self._workers.values() if w.healthy]

    def get_gang_jobs(self, gang_id: str) -> list[ControllerJob]:
        with self._lock:
            job_ids = self._gangs.get(gang_id, set())
            return [self._jobs[jid] for jid in job_ids if jid in self._jobs]
```

**Deliverable**: `ControllerState` class with thread-safe operations.

**Test checkpoint**:
```python
def test_controller_state_fifo_order():
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=..., submitted_at_ms=100)
    job2 = ControllerJob(job_id=JobId("j2"), request=..., submitted_at_ms=200)
    state.add_job(job1)
    state.add_job(job2)

    assert state.pop_next_pending().job_id == "j1"
    assert state.pop_next_pending().job_id == "j2"
    assert state.pop_next_pending() is None

def test_controller_state_skip_non_pending():
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=...)
    job1.state = cluster_pb2.JOB_STATE_RUNNING  # Already started
    job2 = ControllerJob(job_id=JobId("j2"), request=...)
    state.add_job(job1)
    state.add_job(job2)

    # Should skip j1 since it's not PENDING
    assert state.pop_next_pending().job_id == "j2"
```

---

### Stage 3: Worker Registry (Static Configuration)

**Goal**: Load workers from config at startup, track health.

```python
# lib/fluster/src/fluster/cluster/controller/workers.py

from dataclasses import dataclass
import grpc
from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState, ControllerWorker
from fluster.cluster.types import WorkerId

@dataclass
class WorkerConfig:
    """Static worker configuration for v0."""
    worker_id: str
    address: str
    resources: cluster_pb2.ResourceSpec


def load_workers_from_config(
    state: ControllerState,
    workers: list[WorkerConfig],
) -> None:
    """Register workers from static config."""
    import time
    now_ms = int(time.time() * 1000)

    for cfg in workers:
        worker = ControllerWorker(
            worker_id=WorkerId(cfg.worker_id),
            address=cfg.address,
            resources=cfg.resources,
            last_heartbeat_ms=now_ms,
        )
        state.add_worker(worker)


def find_worker_for_job(
    state: ControllerState,
    job: "ControllerJob",
) -> ControllerWorker | None:
    """Find a worker that can run the given job.

    For v0: simple first-fit on available workers.
    Future: resource matching (TPU type, memory, etc).
    """
    workers = state.get_available_workers()
    for worker in workers:
        # TODO: Check resource compatibility
        # For now, any healthy worker works
        return worker
    return None
```

**Deliverable**: Worker loading from config, simple scheduling.

**Test checkpoint**:
```python
def test_load_workers():
    state = ControllerState()
    workers = [
        WorkerConfig("w1", "host1:8080", make_cpu_spec()),
        WorkerConfig("w2", "host2:8080", make_cpu_spec()),
    ]
    load_workers_from_config(state, workers)

    assert len(state.get_available_workers()) == 2
    assert state.get_worker(WorkerId("w1")).address == "host1:8080"
```

---

### Stage 4: Job Scheduler Thread

**Goal**: Background thread that matches pending jobs to workers.

```python
# lib/fluster/src/fluster/cluster/controller/scheduler.py

import threading
import time
import logging
from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState, ControllerJob
from fluster.cluster.controller.workers import find_worker_for_job

logger = logging.getLogger(__name__)

class Scheduler:
    """Background scheduler that dispatches jobs to workers.

    Wakes on:
    - 1 second timer
    - wake() called (new worker, job finished, etc)
    """

    def __init__(
        self,
        state: ControllerState,
        dispatch_fn: "Callable[[ControllerJob, ControllerWorker], bool]",
        interval_seconds: float = 1.0,
    ):
        self._state = state
        self._dispatch_fn = dispatch_fn
        self._interval = interval_seconds
        self._wake_event = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def wake(self) -> None:
        """Signal scheduler to run immediately."""
        self._wake_event.set()

    def _run(self) -> None:
        while not self._stop:
            self._wake_event.wait(timeout=self._interval)
            self._wake_event.clear()

            if self._stop:
                break

            self._schedule_pending_jobs()

    def _schedule_pending_jobs(self) -> None:
        """Try to schedule all pending jobs."""
        while True:
            job = self._state.pop_next_pending()
            if not job:
                break

            worker = find_worker_for_job(self._state, job)
            if not worker:
                # No worker available, re-queue
                self._state.add_job(job)  # Goes to back of queue
                break

            # Dispatch to worker
            success = self._dispatch_fn(job, worker)
            if success:
                job.state = cluster_pb2.JOB_STATE_RUNNING
                job.worker_id = worker.worker_id
                job.started_at_ms = int(time.time() * 1000)
                worker.running_jobs.add(job.job_id)
                logger.info(f"Dispatched job {job.job_id} to worker {worker.worker_id}")
            else:
                # Dispatch failed, mark worker unhealthy and retry
                worker.healthy = False
                self._state.add_job(job)  # Re-queue
                logger.warning(f"Failed to dispatch to {worker.worker_id}, re-queuing job")
```

**Deliverable**: Scheduler that runs dispatch loop.

**Test checkpoint**:
```python
def test_scheduler_dispatches_jobs():
    state = ControllerState()
    dispatched = []

    def mock_dispatch(job, worker):
        dispatched.append((job.job_id, worker.worker_id))
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    # Add worker and job
    state.add_worker(ControllerWorker(WorkerId("w1"), "addr", make_spec()))
    state.add_job(ControllerJob(JobId("j1"), request=...))

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    assert dispatched == [("j1", "w1")]
    assert state.get_job(JobId("j1")).state == cluster_pb2.JOB_STATE_RUNNING

def test_scheduler_requeues_when_no_workers():
    state = ControllerState()
    scheduler = Scheduler(state, lambda j, w: True, interval_seconds=0.1)

    # Add job but no workers
    state.add_job(ControllerJob(JobId("j1"), request=...))

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Job should still be pending (re-queued)
    assert state.get_job(JobId("j1")).state == cluster_pb2.JOB_STATE_PENDING
```

---

### Stage 5: Worker Heartbeat Monitor

**Goal**: Periodically poll workers, detect failures, mark jobs.

```python
# lib/fluster/src/fluster/cluster/controller/heartbeat.py

import threading
import time
import logging
from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState

logger = logging.getLogger(__name__)

class HeartbeatMonitor:
    """Monitors worker health via periodic heartbeats.

    On N consecutive failures:
    - Mark worker unhealthy
    - Mark all running jobs as WORKER_FAILED
    - Trigger retry logic
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        state: ControllerState,
        heartbeat_fn: "Callable[[str], cluster_pb2.HeartbeatResponse | None]",
        on_worker_failed: "Callable[[WorkerId, list[JobId]], None]",
        interval_seconds: float = 1.0,
    ):
        self._state = state
        self._heartbeat_fn = heartbeat_fn
        self._on_worker_failed = on_worker_failed
        self._interval = interval_seconds
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop:
            time.sleep(self._interval)
            self._check_all_workers()

    def _check_all_workers(self) -> None:
        workers = list(self._state._workers.values())  # Snapshot
        now_ms = int(time.time() * 1000)

        for worker in workers:
            if not worker.healthy:
                continue

            response = self._heartbeat_fn(worker.address)

            if response is None:
                # Heartbeat failed
                worker.consecutive_failures += 1
                logger.warning(
                    f"Heartbeat failed for {worker.worker_id} "
                    f"({worker.consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES})"
                )

                if worker.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self._handle_worker_failure(worker)
            else:
                # Success - reset failure count, update job states
                worker.consecutive_failures = 0
                worker.last_heartbeat_ms = now_ms
                self._sync_job_states(worker, response)

    def _handle_worker_failure(self, worker) -> None:
        """Mark worker dead, fail its jobs."""
        logger.error(f"Worker {worker.worker_id} declared dead")
        worker.healthy = False

        failed_jobs = list(worker.running_jobs)
        worker.running_jobs.clear()

        # Mark jobs as failed
        for job_id in failed_jobs:
            job = self._state.get_job(job_id)
            if job:
                job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
                job.finished_at_ms = int(time.time() * 1000)
                job.error = f"Worker {worker.worker_id} failed"

        # Notify for retry handling
        self._on_worker_failed(worker.worker_id, failed_jobs)

    def _sync_job_states(self, worker, response: cluster_pb2.HeartbeatResponse) -> None:
        """Update controller state from worker's heartbeat response."""
        for status in response.jobs:
            job = self._state.get_job(JobId(status.job_id))
            if not job:
                continue

            # Update state from worker
            if status.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                job.state = status.state
                job.finished_at_ms = status.finished_at_ms
                job.error = status.error or None
                job.exit_code = status.exit_code
                worker.running_jobs.discard(job.job_id)
```

**Deliverable**: Heartbeat monitor with failure detection.

**Test checkpoint**:
```python
def test_heartbeat_marks_worker_failed_after_n_failures():
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_spec())
    job = ControllerJob(JobId("j1"), request=...)
    job.state = cluster_pb2.JOB_STATE_RUNNING
    job.worker_id = worker.worker_id
    worker.running_jobs.add(job.job_id)

    state.add_worker(worker)
    state.add_job(job)  # Add directly, not via queue

    failed_workers = []
    def on_failed(wid, jobs):
        failed_workers.append((wid, jobs))

    # Heartbeat always fails
    monitor = HeartbeatMonitor(
        state,
        heartbeat_fn=lambda addr: None,  # Always fail
        on_worker_failed=on_failed,
        interval_seconds=0.05,
    )

    monitor.start()
    time.sleep(0.3)  # Wait for 3+ failures
    monitor.stop()

    assert not worker.healthy
    assert job.state == cluster_pb2.JOB_STATE_WORKER_FAILED
    assert failed_workers == [("w1", ["j1"])]
```

---

### Stage 6: Job Failure and Retry Logic

**Goal**: Re-queue eligible failed jobs, track retry counts.

```python
# lib/fluster/src/fluster/cluster/controller/retry.py

import logging
from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState, ControllerJob
from fluster.cluster.types import JobId

logger = logging.getLogger(__name__)

def handle_job_failure(
    state: ControllerState,
    job_id: JobId,
    is_worker_failure: bool,
) -> bool:
    """Handle a job failure, potentially retrying.

    Returns True if job was re-queued for retry.
    """
    job = state.get_job(job_id)
    if not job:
        return False

    if is_worker_failure:
        job.preemption_count += 1
        can_retry = job.preemption_count <= job.max_retries_preemption
    else:
        job.failure_count += 1
        can_retry = job.failure_count <= job.max_retries_failure

    if can_retry:
        logger.info(
            f"Retrying job {job_id} "
            f"(failures={job.failure_count}, preemptions={job.preemption_count})"
        )
        job.state = cluster_pb2.JOB_STATE_PENDING
        job.worker_id = None
        job.started_at_ms = None
        job.finished_at_ms = None
        job.error = None
        state.add_job(job)  # Re-queue
        return True
    else:
        logger.warning(f"Job {job_id} exceeded retry limit, not retrying")
        return False


def handle_gang_failure(
    state: ControllerState,
    gang_id: str,
    is_worker_failure: bool,
) -> list[JobId]:
    """Handle gang failure - terminate all jobs, optionally retry.

    Returns list of job IDs that were re-queued.
    """
    jobs = state.get_gang_jobs(gang_id)
    if not jobs:
        return []

    # First, terminate all jobs in gang
    for job in jobs:
        if job.state == cluster_pb2.JOB_STATE_RUNNING:
            job.state = cluster_pb2.JOB_STATE_KILLED
            job.error = f"Gang {gang_id} failed"

    # Check if gang can be retried (all jobs must have retries left)
    if is_worker_failure:
        can_retry = all(
            job.preemption_count < job.max_retries_preemption
            for job in jobs
        )
    else:
        can_retry = all(
            job.failure_count < job.max_retries_failure
            for job in jobs
        )

    if can_retry:
        retried = []
        for job in jobs:
            if is_worker_failure:
                job.preemption_count += 1
            else:
                job.failure_count += 1
            job.state = cluster_pb2.JOB_STATE_PENDING
            job.worker_id = None
            state.add_job(job)
            retried.append(job.job_id)
        return retried

    return []
```

**Deliverable**: Retry logic respecting limits.

**Test checkpoint**:
```python
def test_job_retry_on_worker_failure():
    state = ControllerState()
    job = ControllerJob(
        JobId("j1"), request=...,
        max_retries_preemption=2,
    )
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    state.add_job(job)

    # First failure - should retry
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=True)
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.preemption_count == 1

    # Second failure - should retry
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=True)
    assert job.preemption_count == 2

    # Third failure - should NOT retry (exceeded limit)
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    assert not handle_job_failure(state, JobId("j1"), is_worker_failure=True)

def test_gang_all_or_nothing_retry():
    state = ControllerState()
    job1 = ControllerJob(JobId("j1"), request=..., gang_id="g1", max_retries_failure=1)
    job2 = ControllerJob(JobId("j2"), request=..., gang_id="g1", max_retries_failure=0)

    state.add_job(job1)
    state.add_job(job2)
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Gang fails - j2 has 0 retries, so entire gang cannot retry
    retried = handle_gang_failure(state, "g1", is_worker_failure=False)
    assert retried == []
    assert job1.state == cluster_pb2.JOB_STATE_KILLED
    assert job2.state == cluster_pb2.JOB_STATE_KILLED
```

---

### Stage 7: Controller Service Implementation

**Goal**: Wire up RPC handlers using the state and scheduler.

```python
# lib/fluster/src/fluster/cluster/controller/service.py

import time
import uuid
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState, ControllerJob
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.types import JobId


class ControllerServiceImpl:
    """ControllerService RPC implementation."""

    def __init__(self, state: ControllerState, scheduler: Scheduler):
        self._state = state
        self._scheduler = scheduler

    def launch_job(
        self,
        request: cluster_pb2.LaunchJobRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.LaunchJobResponse:
        job_id = str(uuid.uuid4())

        job = ControllerJob(
            job_id=JobId(job_id),
            request=request,
            submitted_at_ms=int(time.time() * 1000),
        )

        self._state.add_job(job)
        self._scheduler.wake()  # Try to schedule immediately

        return cluster_pb2.LaunchJobResponse(job_id=job_id)

    def get_job_status(
        self,
        request: cluster_pb2.GetJobStatusRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.GetJobStatusResponse:
        job = self._state.get_job(JobId(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        return cluster_pb2.GetJobStatusResponse(
            job=cluster_pb2.JobStatus(
                job_id=job.job_id,
                state=job.state,
                error=job.error or "",
                exit_code=job.exit_code or 0,
                started_at_ms=job.started_at_ms or 0,
                finished_at_ms=job.finished_at_ms or 0,
                worker_id=job.worker_id or "",
            )
        )

    def terminate_job(
        self,
        request: cluster_pb2.TerminateJobRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Empty:
        job = self._state.get_job(JobId(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        # TODO: Send kill to worker
        job.state = cluster_pb2.JOB_STATE_KILLED
        job.finished_at_ms = int(time.time() * 1000)

        return cluster_pb2.Empty()

    def list_jobs(
        self,
        request: cluster_pb2.ListJobsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.ListJobsResponse:
        jobs = [
            cluster_pb2.JobStatus(
                job_id=j.job_id,
                state=j.state,
                worker_id=j.worker_id or "",
            )
            for j in self._state._jobs.values()
        ]
        return cluster_pb2.ListJobsResponse(jobs=jobs)
```

**Deliverable**: Working RPC handlers.

**Test checkpoint**:
```python
def test_launch_job_adds_to_queue():
    state = ControllerState()
    scheduler = Scheduler(state, lambda j, w: True)
    service = ControllerServiceImpl(state, scheduler)

    response = service.launch_job(
        cluster_pb2.LaunchJobRequest(name="test"),
        None,
    )

    assert response.job_id
    job = state.get_job(JobId(response.job_id))
    assert job.state == cluster_pb2.JOB_STATE_PENDING
```

---

### Stage 8: Integration Test - End to End

**Goal**: Full flow from job submission to completion.

```python
def test_full_job_lifecycle(worker_server):
    """Integration test with real worker."""
    # Start controller with worker
    state = ControllerState()
    load_workers_from_config(state, [
        WorkerConfig("w1", worker_server.address, make_cpu_spec())
    ])

    # Create dispatcher that calls real worker
    def dispatch(job, worker):
        client = create_worker_client(worker.address)
        response = client.run_job(cluster_pb2.RunJobRequest(
            job_id=job.job_id,
            serialized_entrypoint=job.request.serialized_entrypoint,
            # ...
        ))
        return response.state != cluster_pb2.JOB_STATE_FAILED

    scheduler = Scheduler(state, dispatch)
    heartbeat_monitor = HeartbeatMonitor(
        state,
        heartbeat_fn=lambda addr: do_heartbeat(addr),
        on_worker_failed=lambda wid, jobs: ...,
    )

    service = ControllerServiceImpl(state, scheduler)

    # Start background threads
    scheduler.start()
    heartbeat_monitor.start()

    try:
        # Submit job
        response = service.launch_job(
            cluster_pb2.LaunchJobRequest(
                name="test",
                serialized_entrypoint=cloudpickle.dumps(lambda: print("hello")),
            ),
            None,
        )

        # Wait for completion
        job_id = JobId(response.job_id)
        for _ in range(100):
            job = state.get_job(job_id)
            if job.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
            ):
                break
            time.sleep(0.1)

        assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED
    finally:
        scheduler.stop()
        heartbeat_monitor.stop()
```

---

### Stage 9: Dashboard (HTTP)

**Goal**: Simple HTML dashboard for visibility.

```python
# lib/fluster/src/fluster/cluster/controller/dashboard.py

from aiohttp import web
from fluster.cluster.controller.state import ControllerState

def create_dashboard_app(state: ControllerState) -> web.Application:
    app = web.Application()

    async def index(request):
        workers = state.get_available_workers()
        jobs = list(state._jobs.values())

        html = f"""
        <html>
        <head><title>Fluster Controller</title></head>
        <body>
        <h1>Fluster Controller Dashboard</h1>

        <h2>Workers ({len(workers)} healthy)</h2>
        <table border="1">
        <tr><th>ID</th><th>Address</th><th>Healthy</th><th>Running</th></tr>
        {"".join(f"<tr><td>{w.worker_id}</td><td>{w.address}</td><td>{w.healthy}</td><td>{len(w.running_jobs)}</td></tr>" for w in state._workers.values())}
        </table>

        <h2>Jobs ({len(jobs)} total)</h2>
        <table border="1">
        <tr><th>ID</th><th>State</th><th>Worker</th><th>Error</th></tr>
        {"".join(f"<tr><td>{j.job_id}</td><td>{j.state}</td><td>{j.worker_id or '-'}</td><td>{j.error or '-'}</td></tr>" for j in jobs)}
        </table>
        </body></html>
        """
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/", index)
    return app
```

**Deliverable**: Basic dashboard showing workers and jobs.

---

## Testing Summary

| Stage | Test Focus | Key Assertions |
|-------|------------|----------------|
| 2 | Queue ordering | FIFO order, skip non-pending |
| 3 | Worker loading | Config -> state |
| 4 | Scheduler | Dispatch loop, re-queue on no workers |
| 5 | Heartbeat | Failure detection, job marking |
| 6 | Retry | Limits respected, gang all-or-nothing |
| 7 | RPC | Basic request/response |
| 8 | Integration | Full lifecycle |

## File Structure

```
lib/fluster/src/fluster/cluster/controller/
├── __init__.py
├── state.py        # Stage 2: ControllerState, ControllerJob, ControllerWorker
├── workers.py      # Stage 3: Worker config loading
├── scheduler.py    # Stage 4: Scheduler thread
├── heartbeat.py    # Stage 5: HeartbeatMonitor
├── retry.py        # Stage 6: Retry logic
├── service.py      # Stage 7: RPC handlers
└── dashboard.py    # Stage 9: HTTP dashboard
```
