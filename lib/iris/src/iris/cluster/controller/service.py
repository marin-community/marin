# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller RPC service implementation handling job, task, and worker operations.

The controller expands jobs into tasks at submission time (a job with replicas=N
creates N tasks). Tasks are the unit of scheduling and execution. Job state is
aggregated from task states.
"""

import json
import logging
import uuid
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.controller.bundle_store import BundleStore
from iris.cluster.controller.events import (
    JobCancelledEvent,
    JobSubmittedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.pending_diagnostics import (
    build_job_pending_hints,
    is_active_scale_up_hint,
)
from iris.cluster.controller.scheduler import SchedulingContext
from iris.cluster.controller.state import (
    ControllerEndpoint,
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerWorker,
)
from iris.cluster import task_logging
from iris.cluster.types import JobName, WorkerId
from iris.logging import LogBuffer
from iris.rpc import cluster_pb2, vm_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.rpc.errors import rpc_error_handler
from iris.rpc.proto_utils import job_state_name, task_state_name
from iris.time_utils import Timer, Timestamp

logger = logging.getLogger(__name__)

DEFAULT_TRANSACTION_LIMIT = 50
DEFAULT_MAX_TOTAL_LINES = 10000
_SLOW_STORAGE_READ_MS = 2000

# Maximum bundle size in bytes (25 MB) - matches client-side limit
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024


class AutoscalerProtocol(Protocol):
    """Protocol for autoscaler operations used by ControllerServiceImpl."""

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Get autoscaler status."""
        ...

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get info for a specific VM."""
        ...

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get initialization log for a VM."""
        ...


class StubFactoryProtocol(Protocol):
    """Protocol for getting cached worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerServiceClientSync: ...
    def evict(self, address: str) -> None: ...


class ControllerProtocol(Protocol):
    """Protocol for controller operations used by ControllerServiceImpl."""

    def wake(self) -> None: ...

    def kill_tasks_on_workers(self, task_ids: set[JobName]) -> None: ...

    def create_scheduling_context(self, workers: list[ControllerWorker]) -> SchedulingContext: ...

    def get_job_scheduling_diagnostics(self, job: ControllerJob, context: SchedulingContext) -> str: ...

    @property
    def autoscaler(self) -> AutoscalerProtocol | None: ...

    stub_factory: StubFactoryProtocol


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Args:
        state: Controller state containing jobs, tasks, and workers
        scheduler: Background scheduler for task dispatch (any object with wake() method)
        bundle_prefix: URI prefix for storing bundles (e.g., gs://bucket/path or file:///path).
                      Required for job submission with bundles.
    """

    def __init__(
        self,
        state: ControllerState,
        controller: ControllerProtocol,
        bundle_prefix: str,
        log_buffer: LogBuffer | None = None,
    ):
        self._state = state
        self._controller = controller
        self._bundle_store = BundleStore(bundle_prefix)
        self._log_buffer = log_buffer

    def _get_autoscaler_pending_hints(self) -> dict[str, str]:
        """Build autoscaler-based pending hints keyed by job id."""
        autoscaler = self._controller.autoscaler
        if autoscaler is None:
            return {}
        status = autoscaler.get_status()
        if not status.HasField("last_routing_decision"):
            return {}
        return build_job_pending_hints(status.last_routing_decision)

    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        """Submit a new job to the controller.

        The job is expanded into tasks based on the replicas field
        (defaulting to 1). Each task has ID "/job/.../index".
        """
        with rpc_error_handler("launching job"):
            if not request.name:
                raise ConnectError(Code.INVALID_ARGUMENT, "Job name is required")

            job_id = JobName.from_wire(request.name)

            # Reject submissions if the parent job has already terminated
            if job_id.parent:
                parent_job = self._state.get_job(job_id.parent)
                if parent_job and parent_job.is_finished():
                    raise ConnectError(
                        Code.FAILED_PRECONDITION,
                        f"Cannot submit job: parent job {job_id.parent} has terminated "
                        f"(state={cluster_pb2.JobState.Name(parent_job.state)})",
                    )

            existing_job = self._state.get_job(job_id)
            if existing_job:
                # By default (fail_if_exists=False), replace finished jobs
                if existing_job.is_finished() and not request.fail_if_exists:
                    logger.info(
                        "Replacing finished job %s (state=%s) with new submission",
                        job_id,
                        cluster_pb2.JobState.Name(existing_job.state),
                    )
                    self._state.remove_finished_job(job_id)
                elif existing_job.is_finished():
                    raise ConnectError(
                        Code.ALREADY_EXISTS,
                        f"Job {job_id} already exists (state={cluster_pb2.JobState.Name(existing_job.state)})",
                    )
                else:
                    raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists and is still running")

            # Handle bundle_blob: upload to bundle store, then replace blob
            # with the resulting GCS path (preserving all other fields).
            if request.bundle_blob:
                # Validate bundle size
                bundle_size = len(request.bundle_blob)
                if bundle_size > MAX_BUNDLE_SIZE_BYTES:
                    bundle_size_mb = bundle_size / (1024 * 1024)
                    max_size_mb = MAX_BUNDLE_SIZE_BYTES / (1024 * 1024)
                    raise ConnectError(
                        Code.INVALID_ARGUMENT,
                        f"Bundle size {bundle_size_mb:.1f}MB exceeds maximum {max_size_mb:.0f}MB",
                    )

                bundle_path = self._bundle_store.write_bundle(job_id.to_wire(), request.bundle_blob)

                new_request = cluster_pb2.Controller.LaunchJobRequest()
                new_request.CopyFrom(request)
                new_request.ClearField("bundle_blob")
                new_request.bundle_gcs_path = bundle_path
                request = new_request

            # Submit job via event API
            self._state.handle_event(
                JobSubmittedEvent(
                    job_id=job_id,
                    request=request,
                    timestamp=Timestamp.now(),
                )
            )
            self._controller.wake()

            num_tasks = len(self._state.get_job_tasks(job_id))
            logger.info(f"Job {job_id} submitted with {num_tasks} task(s)")
            return cluster_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())

    def get_job_status(
        self,
        request: cluster_pb2.Controller.GetJobStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        """Get status of a specific job including all task statuses."""
        job = self._state.get_job(JobName.from_wire(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        # Build task statuses with attempts, aggregate counts in single pass
        task_statuses = []
        total_failure_count = 0
        total_preemption_count = 0
        for task in self._state.get_job_tasks(job.job_id):
            total_failure_count += task.failure_count
            total_preemption_count += task.preemption_count

            worker_address = ""
            if task.worker_id:
                worker = self._state.get_worker(task.worker_id)
                if worker:
                    worker_address = worker.address

            # Convert task attempts to proto
            attempts = []
            for attempt in task.attempts:
                proto_attempt = cluster_pb2.TaskAttempt(
                    attempt_id=attempt.attempt_id,
                    worker_id=str(attempt.worker_id) if attempt.worker_id else "",
                    state=attempt.state,
                    exit_code=attempt.exit_code or 0,
                    error=attempt.error or "",
                    is_worker_failure=attempt.is_worker_failure,
                )
                if attempt.started_at is not None:
                    proto_attempt.started_at.CopyFrom(attempt.started_at.to_proto())
                if attempt.finished_at is not None:
                    proto_attempt.finished_at.CopyFrom(attempt.finished_at.to_proto())
                attempts.append(proto_attempt)

            current_attempt = task.current_attempt

            proto_task_status = cluster_pb2.TaskStatus(
                task_id=task.task_id.to_wire(),
                state=task.state,
                worker_id=str(task.worker_id) if task.worker_id else "",
                worker_address=worker_address,
                exit_code=task.exit_code or 0,
                error=task.error or "",
                current_attempt_id=task.current_attempt_id,
                attempts=attempts,
                log_directory=current_attempt.log_directory if current_attempt and current_attempt.log_directory else "",
            )
            if current_attempt and current_attempt.started_at:
                proto_task_status.started_at.CopyFrom(current_attempt.started_at.to_proto())
            if current_attempt and current_attempt.finished_at:
                proto_task_status.finished_at.CopyFrom(current_attempt.finished_at.to_proto())
            task_statuses.append(proto_task_status)

        # Get scheduling diagnostics for pending jobs
        pending_reason = ""
        if job.state == cluster_pb2.JOB_STATE_PENDING:
            # Create scheduling context once for all tasks
            workers = self._state.list_all_workers()
            sched_context = self._controller.create_scheduling_context(workers)
            # Get job-level diagnostics (expensive but only for detail view)
            pending_reason = self._controller.get_job_scheduling_diagnostics(job, sched_context)
            autoscaler_hint = self._get_autoscaler_pending_hints().get(job.job_id.to_wire(), "")
            # Only override scheduler diagnostics when autoscaler is actively
            # requesting new capacity. Otherwise, scheduler root-cause details
            # (e.g., constraint/resource mismatch) are more actionable.
            if autoscaler_hint and is_active_scale_up_hint(autoscaler_hint):
                pending_reason = autoscaler_hint

        # Build the JobStatus proto and set timestamps
        proto_job_status = cluster_pb2.JobStatus(
            job_id=job.job_id.to_wire(),
            state=job.state,
            error=job.error or "",
            exit_code=job.exit_code or 0,
            failure_count=total_failure_count,
            preemption_count=total_preemption_count,
            tasks=task_statuses,
            name=job.request.name if job.request else "",
            pending_reason=pending_reason,
        )
        if job.request:
            proto_job_status.resources.CopyFrom(job.request.resources)
        if job.started_at:
            proto_job_status.started_at.CopyFrom(job.started_at.to_proto())
        if job.finished_at:
            proto_job_status.finished_at.CopyFrom(job.finished_at.to_proto())
        if job.submitted_at:
            proto_job_status.submitted_at.CopyFrom(job.submitted_at.to_proto())

        return cluster_pb2.Controller.GetJobStatusResponse(
            job=proto_job_status,
            request=job.request,
        )

    def terminate_job(
        self,
        request: cluster_pb2.Controller.TerminateJobRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Terminate a running job and all its children.

        Cascade termination is performed depth-first: all children are
        terminated before the parent. All tasks within each job are killed.
        """
        job = self._state.get_job(JobName.from_wire(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        self._terminate_job_tree(JobName.from_wire(request.job_id))
        return cluster_pb2.Empty()

    def _terminate_job_tree(self, job_id: JobName) -> None:
        """Recursively terminate a job and all its descendants (depth-first)."""
        job = self._state.get_job(job_id)
        if not job:
            return

        # First, terminate all children recursively
        children = self._state.get_children(job_id)
        for child in children:
            self._terminate_job_tree(child.job_id)

        if job.is_finished():
            return

        # Cancel the job via event API (this will kill all tasks)
        txn = self._state.handle_event(
            JobCancelledEvent(
                job_id=job_id,
                reason="Terminated by user",
            )
        )

        # Send kill RPCs to workers for any tasks that were killed
        if txn.tasks_to_kill:
            self._controller.kill_tasks_on_workers(txn.tasks_to_kill)

    def list_jobs(
        self,
        request: cluster_pb2.Controller.ListJobsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListJobsResponse:
        """List jobs with server-side pagination, sorting, and filtering."""
        # State priority order for sorting (active states first)
        STATE_ORDER = {
            cluster_pb2.JOB_STATE_RUNNING: 0,
            cluster_pb2.JOB_STATE_BUILDING: 1,
            cluster_pb2.JOB_STATE_PENDING: 2,
            cluster_pb2.JOB_STATE_SUCCEEDED: 3,
            cluster_pb2.JOB_STATE_FAILED: 4,
            cluster_pb2.JOB_STATE_KILLED: 5,
            cluster_pb2.JOB_STATE_WORKER_FAILED: 6,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE: 7,
        }

        # Build all job status objects
        all_jobs: list[cluster_pb2.JobStatus] = []
        name_filter = request.name_filter.lower() if request.name_filter else ""
        state_filter = request.state_filter.lower() if request.state_filter else ""
        autoscaler_pending_hints = self._get_autoscaler_pending_hints()

        for j in self._state.list_all_jobs():
            # Apply name filter
            job_name = j.request.name if j.request else ""
            if name_filter and name_filter not in job_name.lower():
                continue

            # Apply state filter (convert state enum to friendly name)
            job_state_name = cluster_pb2.JobState.Name(j.state).replace("JOB_STATE_", "").lower()
            if state_filter and state_filter != job_state_name:
                continue

            # Aggregate counts from all tasks in single pass
            tasks = self._state.get_job_tasks(j.job_id)
            total_failure_count = 0
            total_preemption_count = 0
            task_state_counts: dict[str, int] = {}
            completed_count = 0

            for task in tasks:
                total_failure_count += task.failure_count
                total_preemption_count += task.preemption_count
                state_name = task_state_name(task.state)
                friendly_name = state_name.replace("TASK_STATE_", "").lower()
                task_state_counts[friendly_name] = task_state_counts.get(friendly_name, 0) + 1
                if state_name in ("TASK_STATE_SUCCEEDED", "TASK_STATE_KILLED"):
                    completed_count += 1

            pending_reason = j.error or ""
            if j.state == cluster_pb2.JOB_STATE_PENDING:
                autoscaler_hint = autoscaler_pending_hints.get(j.job_id.to_wire(), "")
                if autoscaler_hint and is_active_scale_up_hint(autoscaler_hint):
                    pending_reason = autoscaler_hint

            proto_job = cluster_pb2.JobStatus(
                job_id=j.job_id.to_wire(),
                state=j.state,
                error=j.error or "",
                exit_code=j.exit_code or 0,
                failure_count=total_failure_count,
                preemption_count=total_preemption_count,
                name=job_name,
                resources=j.request.resources if j.request else cluster_pb2.ResourceSpecProto(),
                task_state_counts=task_state_counts,
                task_count=len(tasks),
                completed_count=completed_count,
                pending_reason=pending_reason,
            )
            if j.started_at:
                proto_job.started_at.CopyFrom(j.started_at.to_proto())
            if j.finished_at:
                proto_job.finished_at.CopyFrom(j.finished_at.to_proto())
            if j.submitted_at:
                proto_job.submitted_at.CopyFrom(j.submitted_at.to_proto())
            all_jobs.append(proto_job)

        total_count = len(all_jobs)

        # Sorting
        sort_field = request.sort_field or cluster_pb2.Controller.JOB_SORT_FIELD_DATE
        sort_dir = request.sort_direction
        # Default direction: descending for date, ascending for others
        if sort_dir == cluster_pb2.Controller.SORT_DIRECTION_UNSPECIFIED:
            sort_dir = (
                cluster_pb2.Controller.SORT_DIRECTION_DESC
                if sort_field == cluster_pb2.Controller.JOB_SORT_FIELD_DATE
                else cluster_pb2.Controller.SORT_DIRECTION_ASC
            )
        reverse = sort_dir == cluster_pb2.Controller.SORT_DIRECTION_DESC

        def sort_key(job: cluster_pb2.JobStatus):
            if sort_field == cluster_pb2.Controller.JOB_SORT_FIELD_DATE:
                return job.submitted_at.epoch_ms if job.submitted_at.epoch_ms else 0
            elif sort_field == cluster_pb2.Controller.JOB_SORT_FIELD_NAME:
                return job.name.lower()
            elif sort_field == cluster_pb2.Controller.JOB_SORT_FIELD_STATE:
                return STATE_ORDER.get(job.state, 99)
            elif sort_field == cluster_pb2.Controller.JOB_SORT_FIELD_FAILURES:
                return job.failure_count
            elif sort_field == cluster_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS:
                return job.preemption_count
            return job.submitted_at.epoch_ms if job.submitted_at.epoch_ms else 0

        all_jobs.sort(key=sort_key, reverse=reverse)

        # Build parent -> children map to keep families together during pagination
        children_by_parent: dict[str, list[cluster_pb2.JobStatus]] = {}
        job_by_name = {job.name: job for job in all_jobs}

        for job in all_jobs:
            # Extract parent name from hierarchical job name (e.g., "/a/b/c" -> "/a/b")
            if job.name and "/" in job.name:
                last_slash = job.name.rfind("/")
                if last_slash > 0:
                    parent_name = job.name[:last_slash]
                    if parent_name in job_by_name:
                        if parent_name not in children_by_parent:
                            children_by_parent[parent_name] = []
                        children_by_parent[parent_name].append(job)

        # Pagination (limit=0 means return all jobs)
        offset = max(request.offset, 0)
        if request.limit > 0:
            limit = min(request.limit, 500)

            # Include jobs in the requested range
            paginated_jobs = all_jobs[offset : offset + limit]

            # Extend pagination to include all children of any parent in this page
            # This ensures parent-child groups stay together even when pagination would split them
            included_names = {job.name for job in paginated_jobs}
            additional_children = []

            for job in paginated_jobs:
                if job.name in children_by_parent:
                    for child in children_by_parent[job.name]:
                        if child.name not in included_names:
                            additional_children.append(child)
                            included_names.add(child.name)

            paginated_jobs.extend(additional_children)
            has_more = offset + limit < total_count
        else:
            paginated_jobs = all_jobs[offset:]
            has_more = False

        return cluster_pb2.Controller.ListJobsResponse(
            jobs=paginated_jobs,
            total_count=total_count,
            has_more=has_more,
        )

    # --- Task Management ---

    def get_task_status(
        self,
        request: cluster_pb2.Controller.GetTaskStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetTaskStatusResponse:
        """Get status of a specific task."""
        try:
            task_id = JobName.from_wire(request.task_id)
            task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        task = self._state.get_task(task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {task_id} not found")

        # Look up worker address
        worker_address = ""
        if task.worker_id:
            worker = self._state.get_worker(task.worker_id)
            if worker:
                worker_address = worker.address

        current_attempt = task.current_attempt

        proto_task_status = cluster_pb2.TaskStatus(
            task_id=task.task_id.to_wire(),
            state=task.state,
            worker_id=str(task.worker_id) if task.worker_id else "",
            worker_address=worker_address,
            exit_code=task.exit_code or 0,
            error=task.error or "",
            current_attempt_id=task.current_attempt_id,
            log_directory=current_attempt.log_directory if current_attempt and current_attempt.log_directory else "",
        )
        if current_attempt and current_attempt.started_at:
            proto_task_status.started_at.CopyFrom(current_attempt.started_at.to_proto())
        if current_attempt and current_attempt.finished_at:
            proto_task_status.finished_at.CopyFrom(current_attempt.finished_at.to_proto())

        return cluster_pb2.Controller.GetTaskStatusResponse(task=proto_task_status)

    def list_tasks(
        self,
        request: cluster_pb2.Controller.ListTasksRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListTasksResponse:
        """List all tasks, optionally filtered by job_id."""
        job_id = JobName.from_wire(request.job_id) if request.job_id else None
        tasks = []

        if job_id:
            tasks = self._state.get_job_tasks(job_id)
        else:
            for job in self._state.list_all_jobs():
                tasks.extend(self._state.get_job_tasks(job.job_id))

        task_statuses = []
        for task in tasks:
            # Look up worker address
            worker_address = ""
            if task.worker_id:
                worker = self._state.get_worker(task.worker_id)
                if worker:
                    worker_address = worker.address

            # Don't add scheduling diagnostics in list view - too expensive
            # Users should check job detail page for scheduling diagnostics
            pending_reason = ""
            can_be_scheduled = False
            if task.state == cluster_pb2.TASK_STATE_PENDING:
                can_be_scheduled = task.can_be_scheduled()

            # Use attempt timestamps since task-level timestamps are not set
            current_attempt = task.current_attempt

            # Convert task attempts to proto
            attempts = []
            for attempt in task.attempts:
                proto_attempt = cluster_pb2.TaskAttempt(
                    attempt_id=attempt.attempt_id,
                    worker_id=str(attempt.worker_id) if attempt.worker_id else "",
                    state=attempt.state,
                    exit_code=attempt.exit_code or 0,
                    error=attempt.error or "",
                    is_worker_failure=attempt.is_worker_failure,
                )
                if attempt.started_at is not None:
                    proto_attempt.started_at.CopyFrom(attempt.started_at.to_proto())
                if attempt.finished_at is not None:
                    proto_attempt.finished_at.CopyFrom(attempt.finished_at.to_proto())
                attempts.append(proto_attempt)

            proto_task_status = cluster_pb2.TaskStatus(
                task_id=task.task_id.to_wire(),
                state=task.state,
                worker_id=str(task.worker_id) if task.worker_id else "",
                worker_address=worker_address,
                exit_code=task.exit_code or 0,
                error=task.error or "",
                current_attempt_id=task.current_attempt_id,
                pending_reason=pending_reason,
                can_be_scheduled=can_be_scheduled,
                attempts=attempts,
                log_directory=current_attempt.log_directory if current_attempt and current_attempt.log_directory else "",
            )
            if current_attempt and current_attempt.started_at:
                proto_task_status.started_at.CopyFrom(current_attempt.started_at.to_proto())
            if current_attempt and current_attempt.finished_at:
                proto_task_status.finished_at.CopyFrom(current_attempt.finished_at.to_proto())
            if task.resource_usage:
                proto_task_status.resource_usage.CopyFrom(task.resource_usage)
            task_statuses.append(proto_task_status)

        return cluster_pb2.Controller.ListTasksResponse(tasks=task_statuses)

    # --- Worker Management ---

    def register(
        self,
        request: cluster_pb2.Controller.RegisterRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterResponse:
        """One-shot worker registration. Returns worker_id.

        Worker registers once, then waits for heartbeats from the controller.
        """
        with rpc_error_handler("registering worker"):
            # Derive worker_id from vm_address if present, otherwise from address
            worker_id = WorkerId(request.metadata.vm_address or request.address)

            self._state.handle_event(
                WorkerRegisteredEvent(
                    worker_id=worker_id,
                    address=request.address,
                    metadata=request.metadata,
                    timestamp=Timestamp.now(),
                )
            )
            self._controller.wake()

            logger.info("Worker registered: %s at %s", worker_id, request.address)
            return cluster_pb2.Controller.RegisterResponse(
                worker_id=str(worker_id),
                accepted=True,
            )

    def notify_task_update(
        self,
        request: cluster_pb2.Controller.NotifyTaskUpdateRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Hint from worker that it has new completions. Triggers priority heartbeat.

        This is a lightweight ping - the actual completion data comes via the next
        heartbeat response.
        """
        # Just wake the scheduler; it will trigger a priority heartbeat for this worker
        self._controller.wake()
        return cluster_pb2.Empty()

    def list_workers(
        self,
        request: cluster_pb2.Controller.ListWorkersRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListWorkersResponse:
        """List all workers with their running task counts."""
        workers = []
        for w in self._state.list_all_workers():
            # Generate status message for unhealthy workers
            status_message = ""
            if not w.healthy:
                if w.consecutive_failures > 0:
                    time_since_last_hb = w.last_heartbeat.age_ms()
                    time_since_str = f"{time_since_last_hb // 1000}s ago" if time_since_last_hb else "never"
                    status_message = f"Heartbeat timeout ({w.consecutive_failures} failures, last seen {time_since_str})"
                else:
                    status_message = "Unhealthy (no failures recorded)"

            workers.append(
                cluster_pb2.Controller.WorkerHealthStatus(
                    worker_id=w.worker_id,
                    healthy=w.healthy,
                    consecutive_failures=w.consecutive_failures,
                    last_heartbeat=w.last_heartbeat.to_proto(),
                    running_job_ids=[task_id.to_wire() for task_id in w.running_tasks],
                    address=w.address,
                    metadata=w.metadata,
                    status_message=status_message,
                )
            )
        return cluster_pb2.Controller.ListWorkersResponse(workers=workers)

    # --- Endpoint Management ---

    def register_endpoint(
        self,
        request: cluster_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterEndpointResponse:
        """Register a service endpoint.

        Endpoints are registered regardless of job state, but only become visible to clients
        (via lookup/list) when the job is executing (not in a terminal state).
        """
        with rpc_error_handler("registering endpoint"):
            endpoint_id = str(uuid.uuid4())

            job = self._state.get_job(JobName.from_wire(request.job_id))
            if not job:
                raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

            endpoint = ControllerEndpoint(
                endpoint_id=endpoint_id,
                name=request.name,
                address=request.address,
                job_id=JobName.from_wire(request.job_id),
                metadata=dict(request.metadata),
                registered_at=Timestamp.now(),
            )

            self._state.add_endpoint(endpoint)

            return cluster_pb2.Controller.RegisterEndpointResponse(endpoint_id=endpoint_id)

    def unregister_endpoint(
        self,
        request: cluster_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Unregister a service endpoint. Idempotent."""
        self._state.remove_endpoint(request.endpoint_id)
        return cluster_pb2.Empty()

    def lookup_endpoint(
        self,
        request: cluster_pb2.Controller.LookupEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.LookupEndpointResponse:
        """Look up a service endpoint by name. Only endpoints for executing jobs are returned."""
        endpoints = self._state.lookup_endpoints(request.name)
        if not endpoints:
            logger.debug("Endpoint lookup found no results: name=%s", request.name)
            return cluster_pb2.Controller.LookupEndpointResponse()

        e = endpoints[0]
        return cluster_pb2.Controller.LookupEndpointResponse(
            endpoint=cluster_pb2.Controller.Endpoint(
                endpoint_id=e.endpoint_id,
                name=e.name,
                address=e.address,
                job_id=e.job_id.to_wire(),
                metadata=e.metadata,
            )
        )

    def list_endpoints(
        self,
        request: cluster_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListEndpointsResponse:
        """List endpoints by name prefix. Only endpoints for executing jobs are returned."""
        endpoints = self._state.list_endpoints_by_prefix(request.prefix)
        return cluster_pb2.Controller.ListEndpointsResponse(
            endpoints=[
                cluster_pb2.Controller.Endpoint(
                    endpoint_id=e.endpoint_id,
                    name=e.name,
                    address=e.address,
                    job_id=e.job_id.to_wire(),
                    metadata=e.metadata,
                )
                for e in endpoints
            ]
        )

    # --- Autoscaler ---

    def get_autoscaler_status(
        self,
        request: cluster_pb2.Controller.GetAutoscalerStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetAutoscalerStatusResponse:
        """Get current autoscaler status with worker info populated."""
        autoscaler = self._controller.autoscaler
        if not autoscaler:
            return cluster_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())

        status = autoscaler.get_status()

        # Build a map of VM address -> worker info for enriching VmInfo
        # Workers register with vm_address in metadata which matches VM's address
        workers = self._state.list_all_workers()
        vm_address_to_worker: dict[str, tuple[str, bool]] = {}
        for w in workers:
            # worker_id is derived from vm_address, use it to match
            if w.metadata and w.metadata.vm_address:
                vm_address_to_worker[w.metadata.vm_address] = (w.worker_id, w.healthy)
            elif w.address:
                # Fallback: match by worker address (without port) if no vm_address
                host = w.address.split(":")[0] if ":" in w.address else w.address
                vm_address_to_worker[host] = (w.worker_id, w.healthy)

        # Enrich VmInfo objects with worker information
        for group in status.groups:
            for slice_info in group.slices:
                for vm in slice_info.vms:
                    if vm.address:
                        worker_info = vm_address_to_worker.get(vm.address)
                        if worker_info:
                            vm.worker_id = worker_info[0]
                            vm.worker_healthy = worker_info[1]

        return cluster_pb2.Controller.GetAutoscalerStatusResponse(status=status)

    # --- VM Logs ---

    # --- Task/Job Logs (batch fetching) ---

    def get_task_logs(
        self,
        request: cluster_pb2.Controller.GetTaskLogsRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.Controller.GetTaskLogsResponse:
        """Get logs for a task or all tasks in a job from storage.

        Fetches logs from durable storage (GCS) instead of proxying from workers.
        This enables post-mortem access to logs after worker VMs terminate.

        If request.id ends in a numeric index, treat as single task.
        Otherwise treat as job ID and fetch logs from all tasks.

        When attempt_id is specified (>= 0), fetches logs only from that specific attempt.
        """
        job_name = JobName.from_wire(request.id)
        max_lines = request.max_total_lines if request.max_total_lines > 0 else DEFAULT_MAX_TOTAL_LINES
        requested_attempt_id = request.attempt_id

        # Detect if this is a task ID (ends in /N) or job ID and collect tasks.
        # When include_children is set, also collect child job statuses so the
        # client can detect state transitions without a separate ListJobs RPC.
        child_job_statuses: list[cluster_pb2.JobStatus] = []
        if job_name.is_task:
            task = self._state.get_task(job_name)
            tasks = [task] if task else []
        else:
            tasks: list[ControllerTask] = []
            prefix = job_name.to_wire()
            if request.include_children:
                for job in self._state.list_all_jobs():
                    job_wire = job.job_id.to_wire()
                    if job_wire == prefix or job_wire.startswith(prefix + "/"):
                        tasks.extend(self._state.get_job_tasks(job.job_id))
                        if job_wire != prefix:
                            child_status = cluster_pb2.JobStatus(
                                job_id=job_wire,
                                state=job.state,
                                exit_code=job.exit_code or 0,
                                error=job.error or "",
                            )
                            if job.finished_at:
                                child_status.finished_at.CopyFrom(job.finished_at.to_proto())
                            child_job_statuses.append(child_status)
            else:
                tasks.extend(self._state.get_job_tasks(job_name))

        # Fetch logs from storage for each task/attempt
        task_logs: list[cluster_pb2.Controller.TaskLogBatch] = []
        total_lines = 0
        truncated = False
        last_timestamp_ms = request.since_ms

        for task in tasks:
            task_id_wire = task.task_id.to_wire()

            # Determine which attempts to fetch
            attempts_to_fetch = []
            if requested_attempt_id >= 0:
                # Specific attempt requested
                if requested_attempt_id >= len(task.attempts):
                    task_logs.append(
                        cluster_pb2.Controller.TaskLogBatch(
                            task_id=task_id_wire,
                            error=f"Attempt {requested_attempt_id} not found (task has {len(task.attempts)} attempts)",
                        )
                    )
                    continue
                attempts_to_fetch = [task.attempts[requested_attempt_id]]
            else:
                # All attempts
                attempts_to_fetch = task.attempts

            # Fetch logs for each attempt
            for attempt in attempts_to_fetch:
                if not attempt.worker_id:
                    task_logs.append(
                        cluster_pb2.Controller.TaskLogBatch(
                            task_id=task_id_wire,
                            error=f"Attempt {attempt.attempt_id} has no assigned worker",
                        )
                    )
                    continue

                if not attempt.log_directory:
                    task_logs.append(
                        cluster_pb2.Controller.TaskLogBatch(
                            task_id=task_id_wire,
                            worker_id=str(attempt.worker_id),
                            error=(
                                f"Attempt {attempt.attempt_id} has no log directory "
                                "(worker may not have reported it yet)"
                            ),
                        )
                    )
                    continue

                try:
                    storage_timer = Timer()
                    reader = task_logging.LogReader.from_log_directory(log_directory=attempt.log_directory)
                    log_entries = reader.read_logs(
                        source=None,  # All sources
                        regex_filter=request.regex if request.regex else None,
                        max_lines=max(0, max_lines - total_lines) if max_lines > 0 else 0,
                    )
                    storage_elapsed = storage_timer.elapsed_ms()
                    if storage_elapsed > _SLOW_STORAGE_READ_MS:
                        logger.warning(
                            "Storage read for %s attempt %d: %dms (slow)",
                            task_id_wire,
                            attempt.attempt_id,
                            storage_elapsed,
                        )

                    worker_logs = []
                    for entry in log_entries:
                        # Filter by timestamp if requested
                        if request.since_ms > 0 and entry.timestamp.epoch_ms <= request.since_ms:
                            continue

                        worker_logs.append(entry)

                        if entry.timestamp.epoch_ms > last_timestamp_ms:
                            last_timestamp_ms = entry.timestamp.epoch_ms

                    batch = cluster_pb2.Controller.TaskLogBatch(
                        task_id=task_id_wire,
                        worker_id=str(attempt.worker_id),
                        logs=worker_logs,
                    )
                    task_logs.append(batch)

                    total_lines += len(worker_logs)
                    if max_lines > 0 and total_lines >= max_lines:
                        truncated = True
                        break

                except Exception as e:
                    task_logs.append(
                        cluster_pb2.Controller.TaskLogBatch(
                            task_id=task_id_wire,
                            worker_id=str(attempt.worker_id),
                            error=f"Failed to fetch logs from storage: {e}",
                        )
                    )

            if truncated:
                break

        return cluster_pb2.Controller.GetTaskLogsResponse(
            task_logs=task_logs,
            last_timestamp_ms=last_timestamp_ms,
            truncated=truncated,
            child_job_statuses=child_job_statuses,
        )

    # --- Profiling ---

    def profile_task(
        self,
        request: cluster_pb2.ProfileTaskRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.ProfileTaskResponse:
        """Profile a running task by proxying to its worker."""
        with rpc_error_handler("profile_task"):
            try:
                task_name = JobName.from_wire(request.task_id)
                task_name.require_task()
            except ValueError as exc:
                raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
            task = self._state.get_task(task_name)
            if not task:
                raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

            worker_id = task.worker_id
            if not worker_id:
                raise ConnectError(Code.FAILED_PRECONDITION, f"Task {request.task_id} not assigned to a worker")

            worker = self._state.get_worker(worker_id)
            if not worker or not worker.healthy:
                raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")

            timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
            stub = self._controller.stub_factory.get_stub(worker.address)
            resp = stub.profile_task(request, timeout_ms=timeout_ms)
            return cluster_pb2.ProfileTaskResponse(
                profile_data=resp.profile_data,
                error=resp.error,
            )

    # --- Transactions ---

    def get_transactions(
        self,
        request: cluster_pb2.Controller.GetTransactionsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetTransactionsResponse:
        """Get recent controller actions for the dashboard action log."""
        limit = request.limit if request.limit > 0 else DEFAULT_TRANSACTION_LIMIT
        transactions = self._state.get_transactions(limit=limit)
        actions = []
        for txn in transactions:
            for action in txn.actions:
                details_str = json.dumps(action.details) if action.details else ""
                proto_action = cluster_pb2.Controller.TransactionAction(
                    action=action.action,
                    entity_id=action.entity_id,
                    details=details_str,
                )
                proto_action.timestamp.CopyFrom(action.timestamp.to_proto())
                actions.append(proto_action)
        return cluster_pb2.Controller.GetTransactionsResponse(actions=actions)

    # --- Cluster Summary ---

    def get_cluster_summary(
        self,
        request: cluster_pb2.Controller.GetClusterSummaryRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetClusterSummaryResponse:
        """Lightweight cluster-wide counters for the dashboard header."""
        jobs = self._state.list_all_jobs()
        state_counts: dict[str, int] = {}
        for job in jobs:
            name = job_state_name(job.state)
            key = name.removeprefix("JOB_STATE_").lower()
            state_counts[key] = state_counts.get(key, 0) + 1

        workers = self._state.list_all_workers()
        healthy = sum(1 for w in workers if w.healthy)

        return cluster_pb2.Controller.GetClusterSummaryResponse(
            job_state_counts=state_counts,
            total_jobs=len(jobs),
            total_workers=len(workers),
            healthy_workers=healthy,
        )

    # --- Process Logs ---

    def get_process_logs(
        self,
        request: cluster_pb2.Controller.GetProcessLogsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetProcessLogsResponse:
        """Get controller process logs from the in-memory ring buffer."""
        if not self._log_buffer:
            return cluster_pb2.Controller.GetProcessLogsResponse(records=[])
        prefix = request.prefix or None
        limit = request.limit if request.limit > 0 else 200
        records = self._log_buffer.query(prefix=prefix, limit=limit)
        return cluster_pb2.Controller.GetProcessLogsResponse(
            records=[
                cluster_pb2.ProcessLogRecord(
                    timestamp=r.timestamp,
                    level=r.level,
                    logger_name=r.logger_name,
                    message=r.message,
                )
                for r in records
            ]
        )

    # --- Worker Detail (unified VM + worker view) ---

    def get_worker_status(
        self,
        request: cluster_pb2.Controller.GetWorkerStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetWorkerStatusResponse:
        """Return unified VM + worker detail for a single worker.

        Resolves the identifier against both VM IDs and worker IDs, then
        returns VM info, worker health, bootstrap logs, and worker daemon
        logs in a single response.
        """
        with rpc_error_handler("get_worker_status"):
            if not request.id:
                raise ConnectError(Code.INVALID_ARGUMENT, "id is required")

            identifier = request.id
            autoscaler = self._controller.autoscaler
            vm_info: vm_pb2.VmInfo | None = None
            scale_group = ""
            bootstrap_logs = ""
            worker: Any = None  # ControllerWorker

            # Try to resolve as VM ID first
            if autoscaler:
                vm_info = autoscaler.get_vm(identifier)
                if vm_info:
                    scale_group = vm_info.scale_group
                    bootstrap_logs = autoscaler.get_init_log(identifier, tail=500)
                    # Enrich with worker info
                    if vm_info.address:
                        for w in self._state.list_all_workers():
                            vm_addr = w.metadata.vm_address if w.metadata else None
                            w_host = w.address.split(":")[0] if w.address and ":" in w.address else w.address
                            if vm_addr == vm_info.address or w_host == vm_info.address:
                                worker = w
                                vm_info.worker_id = w.worker_id
                                vm_info.worker_healthy = w.healthy
                                break

            # If not found as VM, try as worker ID
            if not vm_info:
                worker = self._state.get_worker(identifier)
                # If we found a worker, scan autoscaler status for its VM
                if worker and autoscaler:
                    status = autoscaler.get_status()
                    found = False
                    for group in status.groups:
                        for slice_info in group.slices:
                            for vm in slice_info.vms:
                                if vm.address and worker.address:
                                    w_host = worker.address.split(":")[0] if ":" in worker.address else worker.address
                                    vm_addr = worker.metadata.vm_address if worker.metadata else None
                                    if vm.address == vm_addr or vm.address == w_host:
                                        vm_info = vm
                                        scale_group = group.name
                                        bootstrap_logs = autoscaler.get_init_log(vm.vm_id, tail=500)
                                        vm_info.worker_id = worker.worker_id
                                        vm_info.worker_healthy = worker.healthy
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break

            if not vm_info and not worker:
                raise ConnectError(Code.NOT_FOUND, f"No VM or worker found for '{identifier}'")

            # Build worker health status proto
            worker_health: cluster_pb2.Controller.WorkerHealthStatus | None = None
            if worker:
                status_message = ""
                if not worker.healthy:
                    if worker.consecutive_failures > 0:
                        age = worker.last_heartbeat.age_ms()
                        status_message = (
                            f"Heartbeat timeout ({worker.consecutive_failures} failures, last seen {age // 1000}s ago)"
                        )
                    else:
                        status_message = "Unhealthy (no failures recorded)"

                worker_health = cluster_pb2.Controller.WorkerHealthStatus(
                    worker_id=worker.worker_id,
                    healthy=worker.healthy,
                    consecutive_failures=worker.consecutive_failures,
                    last_heartbeat=worker.last_heartbeat.to_proto(),
                    running_job_ids=[tid.to_wire() for tid in worker.running_tasks],
                    address=worker.address,
                    metadata=worker.metadata,
                    status_message=status_message,
                )

            # Fetch worker daemon logs if worker is healthy
            worker_logs: list[cluster_pb2.ProcessLogRecord] = []
            if worker and worker.healthy:
                try:
                    stub = self._controller.stub_factory.get_stub(worker.address)
                    resp = stub.get_process_logs(
                        cluster_pb2.Worker.GetProcessLogsRequest(limit=200),
                        timeout_ms=10000,
                    )
                    worker_logs = list(resp.records)
                except Exception:
                    logger.debug("Failed to fetch worker logs for %s", identifier, exc_info=True)

            # Collect recent task history for this worker
            recent_tasks: list[cluster_pb2.TaskStatus] = []
            if worker:
                tasks = self._state.get_tasks_for_worker(worker.worker_id, limit=50)
                for task in tasks:
                    current_attempt = task.current_attempt
                    entry = cluster_pb2.TaskStatus(
                        task_id=task.task_id.to_wire(),
                        state=task.state,
                        worker_id=str(task.worker_id) if task.worker_id else "",
                        exit_code=task.exit_code or 0,
                        error=task.error or "",
                        current_attempt_id=task.current_attempt_id,
                        log_directory=(
                            current_attempt.log_directory if current_attempt and current_attempt.log_directory else ""
                        ),
                    )
                    if current_attempt and current_attempt.started_at:
                        entry.started_at.CopyFrom(current_attempt.started_at.to_proto())
                    if current_attempt and current_attempt.finished_at:
                        entry.finished_at.CopyFrom(current_attempt.finished_at.to_proto())
                    if task.resource_usage:
                        entry.resource_usage.CopyFrom(task.resource_usage)
                    recent_tasks.append(entry)

            resp = cluster_pb2.Controller.GetWorkerStatusResponse(
                bootstrap_logs=bootstrap_logs,
                scale_group=scale_group,
                worker_logs=worker_logs,
                recent_tasks=recent_tasks,
            )
            if vm_info:
                resp.vm.CopyFrom(vm_info)
            if worker_health:
                resp.worker.CopyFrom(worker_health)
            if worker:
                if worker.resource_snapshot:
                    resp.current_resources.CopyFrom(worker.resource_snapshot)
                for snapshot in worker.resource_history:
                    resp.resource_history.append(snapshot)
            return resp
