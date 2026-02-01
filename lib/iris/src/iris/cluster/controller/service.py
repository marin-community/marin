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

from iris.cluster.controller.bundle_store import BundleStore
from iris.cluster.controller.events import (
    JobCancelledEvent,
    JobSubmittedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.scheduler import SchedulingContext, TaskScheduleResult
from iris.cluster.controller.state import ControllerEndpoint, ControllerState, ControllerTask
from iris.cluster.types import JobId, TaskId, WorkerId
from iris.logging import LogBuffer
from iris.rpc import cluster_pb2, vm_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.rpc.errors import rpc_error_handler
from iris.rpc.proto_utils import task_state_name
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)

DEFAULT_TRANSACTION_LIMIT = 50


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


class SchedulerProtocol(Protocol):
    """Protocol for scheduler operations used by ControllerServiceImpl."""

    def wake(self) -> None: ...

    def kill_tasks_on_workers(self, task_ids: set[TaskId]) -> None:
        """Send KILL RPCs to workers for tasks that were running."""
        ...

    def task_schedule_status(self, task: ControllerTask, context: SchedulingContext) -> TaskScheduleResult:
        """Get the current scheduling status of a task (for dashboard display)."""
        ...

    @property
    def autoscaler(self) -> AutoscalerProtocol | None:
        """Get the autoscaler instance, if autoscaling is enabled."""
        ...


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
        scheduler: SchedulerProtocol,
        bundle_prefix: str,
        log_buffer: LogBuffer | None = None,
    ):
        self._state = state
        self._scheduler = scheduler
        self._bundle_store = BundleStore(bundle_prefix)
        self._log_buffer = log_buffer

    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        """Submit a new job to the controller.

        The job is expanded into tasks based on the replicas field
        (defaulting to 1). Each task has ID "{job_id}/task-{index}".
        """
        with rpc_error_handler("launching job"):
            if not request.name:
                raise ConnectError(Code.INVALID_ARGUMENT, "Job name is required")

            job_id = request.name

            if self._state.get_job(JobId(job_id)):
                raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists")

            # Handle bundle_blob: upload to bundle store, then replace blob
            # with the resulting GCS path (preserving all other fields).
            if request.bundle_blob:
                bundle_path = self._bundle_store.write_bundle(job_id, request.bundle_blob)

                new_request = cluster_pb2.Controller.LaunchJobRequest()
                new_request.CopyFrom(request)
                new_request.ClearField("bundle_blob")
                new_request.bundle_gcs_path = bundle_path
                request = new_request

            # Submit job via event API
            self._state.handle_event(
                JobSubmittedEvent(
                    job_id=JobId(job_id),
                    request=request,
                    timestamp_ms=now_ms(),
                )
            )
            self._scheduler.wake()

            num_tasks = len(self._state.get_job_tasks(JobId(job_id)))
            logger.info(f"Job {job_id} submitted with {num_tasks} task(s)")
            return cluster_pb2.Controller.LaunchJobResponse(job_id=job_id)

    def get_job_status(
        self,
        request: cluster_pb2.Controller.GetJobStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        """Get status of a specific job including all task statuses."""
        job = self._state.get_job(JobId(request.job_id))
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
            attempts = [
                cluster_pb2.TaskAttempt(
                    attempt_id=attempt.attempt_id,
                    worker_id=str(attempt.worker_id) if attempt.worker_id else "",
                    state=attempt.state,
                    exit_code=attempt.exit_code or 0,
                    error=attempt.error or "",
                    started_at_ms=attempt.started_at_ms or 0,
                    finished_at_ms=attempt.finished_at_ms or 0,
                    is_worker_failure=attempt.is_worker_failure,
                )
                for attempt in task.attempts
            ]

            current_attempt = task.current_attempt
            task_started_at_ms = current_attempt.started_at_ms or 0 if current_attempt else 0
            task_finished_at_ms = current_attempt.finished_at_ms or 0 if current_attempt else 0

            task_statuses.append(
                cluster_pb2.TaskStatus(
                    task_id=str(task.task_id),
                    job_id=str(task.job_id),
                    task_index=task.task_index,
                    state=task.state,
                    worker_id=str(task.worker_id) if task.worker_id else "",
                    worker_address=worker_address,
                    started_at_ms=task_started_at_ms,
                    finished_at_ms=task_finished_at_ms,
                    exit_code=task.exit_code or 0,
                    error=task.error or "",
                    current_attempt_id=task.current_attempt_id,
                    attempts=attempts,
                )
            )

        return cluster_pb2.Controller.GetJobStatusResponse(
            job=cluster_pb2.JobStatus(
                job_id=job.job_id,
                state=job.state,
                error=job.error or "",
                exit_code=job.exit_code or 0,
                started_at_ms=job.started_at_ms or 0,
                finished_at_ms=job.finished_at_ms or 0,
                parent_job_id=str(job.parent_job_id) if job.parent_job_id else "",
                failure_count=total_failure_count,
                preemption_count=total_preemption_count,
                tasks=task_statuses,
            )
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
        job = self._state.get_job(JobId(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        self._terminate_job_tree(JobId(request.job_id))
        return cluster_pb2.Empty()

    def _terminate_job_tree(self, job_id: JobId) -> None:
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
            self._scheduler.kill_tasks_on_workers(txn.tasks_to_kill)

    def list_jobs(
        self,
        request: cluster_pb2.Controller.ListJobsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListJobsResponse:
        """List all jobs."""
        jobs = []
        for j in self._state.list_all_jobs():
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

            # Job-level diagnostic uses the job error; per-task scheduling
            # diagnostics are computed in list_tasks where they belong.
            pending_reason = j.error or ""

            jobs.append(
                cluster_pb2.JobStatus(
                    job_id=j.job_id,
                    state=j.state,
                    error=j.error or "",
                    exit_code=j.exit_code or 0,
                    started_at_ms=j.started_at_ms or 0,
                    finished_at_ms=j.finished_at_ms or 0,
                    parent_job_id=str(j.parent_job_id) if j.parent_job_id else "",
                    failure_count=total_failure_count,
                    preemption_count=total_preemption_count,
                    name=j.request.name if j.request else "",
                    submitted_at_ms=j.submitted_at_ms or 0,
                    resources=j.request.resources if j.request else cluster_pb2.ResourceSpecProto(),
                    task_state_counts=task_state_counts,
                    task_count=len(tasks),
                    completed_count=completed_count,
                    pending_reason=pending_reason,
                )
            )
        return cluster_pb2.Controller.ListJobsResponse(jobs=jobs)

    # --- Task Management ---

    def get_task_status(
        self,
        request: cluster_pb2.Controller.GetTaskStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetTaskStatusResponse:
        """Get status of a specific task."""
        task_id = TaskId(f"{request.job_id}/task-{request.task_index}")
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
        started_at_ms = current_attempt.started_at_ms or 0 if current_attempt else 0
        finished_at_ms = current_attempt.finished_at_ms or 0 if current_attempt else 0

        return cluster_pb2.Controller.GetTaskStatusResponse(
            task=cluster_pb2.TaskStatus(
                task_id=str(task.task_id),
                job_id=str(task.job_id),
                task_index=task.task_index,
                state=task.state,
                worker_id=str(task.worker_id) if task.worker_id else "",
                worker_address=worker_address,
                started_at_ms=started_at_ms,
                finished_at_ms=finished_at_ms,
                exit_code=task.exit_code or 0,
                error=task.error or "",
                current_attempt_id=task.current_attempt_id,
            )
        )

    def list_tasks(
        self,
        request: cluster_pb2.Controller.ListTasksRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListTasksResponse:
        """List all tasks, optionally filtered by job_id."""
        job_id = JobId(request.job_id) if request.job_id else None
        tasks = []

        if job_id:
            tasks = self._state.get_job_tasks(job_id)
        else:
            for job in self._state.list_all_jobs():
                tasks.extend(self._state.get_job_tasks(job.job_id))

        # Build scheduling context once for all pending-task diagnostics
        workers = self._state.get_available_workers()
        sched_context = SchedulingContext.from_workers(workers)

        task_statuses = []
        for task in tasks:
            # Look up worker address
            worker_address = ""
            if task.worker_id:
                worker = self._state.get_worker(task.worker_id)
                if worker:
                    worker_address = worker.address

            # Add scheduling diagnostics for pending tasks
            pending_reason = ""
            can_be_scheduled = False
            if task.state == cluster_pb2.TASK_STATE_PENDING:
                can_be_scheduled = task.can_be_scheduled()
                schedule_status = self._scheduler.task_schedule_status(task, sched_context)
                pending_reason = schedule_status.failure_reason or ""

            # Use attempt timestamps since task-level timestamps are not set
            current_attempt = task.current_attempt
            started_at_ms = current_attempt.started_at_ms or 0 if current_attempt else 0
            finished_at_ms = current_attempt.finished_at_ms or 0 if current_attempt else 0

            task_statuses.append(
                cluster_pb2.TaskStatus(
                    task_id=str(task.task_id),
                    job_id=str(task.job_id),
                    task_index=task.task_index,
                    state=task.state,
                    worker_id=str(task.worker_id) if task.worker_id else "",
                    worker_address=worker_address,
                    started_at_ms=started_at_ms,
                    finished_at_ms=finished_at_ms,
                    exit_code=task.exit_code or 0,
                    error=task.error or "",
                    current_attempt_id=task.current_attempt_id,
                    pending_reason=pending_reason,
                    can_be_scheduled=can_be_scheduled,
                )
            )

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
                    timestamp_ms=now_ms(),
                )
            )
            self._scheduler.wake()

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
        self._scheduler.wake()
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
                    time_since_last_hb = now_ms() - w.last_heartbeat_ms if w.last_heartbeat_ms else 0
                    time_since_str = f"{time_since_last_hb // 1000}s ago" if time_since_last_hb else "never"
                    status_message = f"Heartbeat timeout ({w.consecutive_failures} failures, last seen {time_since_str})"
                else:
                    status_message = "Unhealthy (no failures recorded)"

            workers.append(
                cluster_pb2.Controller.WorkerHealthStatus(
                    worker_id=w.worker_id,
                    healthy=w.healthy,
                    consecutive_failures=w.consecutive_failures,
                    last_heartbeat_ms=w.last_heartbeat_ms,
                    running_job_ids=list(w.running_tasks),  # Now contains task IDs
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

            job = self._state.get_job(JobId(request.job_id))
            if not job:
                raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

            endpoint = ControllerEndpoint(
                endpoint_id=endpoint_id,
                name=request.name,
                address=request.address,
                job_id=JobId(request.job_id),
                metadata=dict(request.metadata),
                registered_at_ms=now_ms(),
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
                job_id=e.job_id,
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
                    job_id=e.job_id,
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
        """Get current autoscaler status."""
        autoscaler = self._scheduler.autoscaler
        if not autoscaler:
            return cluster_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())

        return cluster_pb2.Controller.GetAutoscalerStatusResponse(status=autoscaler.get_status())

    # --- VM Logs ---

    def get_vm_logs(
        self,
        request: cluster_pb2.Controller.GetVmLogsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetVmLogsResponse:
        """Get initialization logs for a VM."""
        autoscaler = self._scheduler.autoscaler
        if not autoscaler:
            raise ConnectError(Code.UNAVAILABLE, "Autoscaler not configured")

        vm_info = autoscaler.get_vm(request.vm_id)
        if not vm_info:
            raise ConnectError(Code.NOT_FOUND, f"VM {request.vm_id} not found")

        tail = request.tail if request.tail > 0 else None
        logs = autoscaler.get_init_log(request.vm_id, tail)

        return cluster_pb2.Controller.GetVmLogsResponse(
            logs=logs,
            vm_id=vm_info.vm_id,
            state=vm_info.state,
        )

    # --- Task Logs (proxied to worker) ---

    def get_task_logs(
        self,
        request: cluster_pb2.Controller.GetTaskLogsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetTaskLogsResponse:
        """Get task logs by proxying the request to the worker that owns the task."""
        job_id = JobId(request.job_id)
        tasks = self._state.get_job_tasks(job_id)
        task = next((t for t in tasks if t.task_index == request.task_index), None)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.job_id}/task-{request.task_index} not found")

        if not task.worker_id:
            raise ConnectError(Code.FAILED_PRECONDITION, "Task has no assigned worker")

        worker = self._state.get_worker(task.worker_id)
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {task.worker_id} not found")

        log_filter = cluster_pb2.Worker.FetchLogsFilter(
            start_ms=request.start_ms,
            max_lines=request.limit,
        )
        worker_client = WorkerServiceClientSync(f"http://{worker.address}")
        worker_resp = worker_client.fetch_task_logs(
            cluster_pb2.Worker.FetchTaskLogsRequest(
                task_id=str(task.task_id),
                filter=log_filter,
            )
        )
        return cluster_pb2.Controller.GetTaskLogsResponse(
            logs=list(worker_resp.logs),
            worker_address=worker.address,
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
                actions.append(
                    cluster_pb2.Controller.TransactionAction(
                        timestamp_ms=action.timestamp_ms,
                        action=action.action,
                        entity_id=action.entity_id,
                        details=details_str,
                    )
                )
        return cluster_pb2.Controller.GetTransactionsResponse(actions=actions)

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
