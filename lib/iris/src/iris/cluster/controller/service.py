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

import logging
import uuid
from pathlib import Path
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.controller.events import (
    JobCancelledEvent,
    JobSubmittedEvent,
    TaskStateChangedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.state import ControllerEndpoint, ControllerState
from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2
from iris.rpc.errors import rpc_error_handler
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)


class SchedulerProtocol(Protocol):
    """Protocol for scheduler operations used by ControllerServiceImpl."""

    def wake(self) -> None: ...

    def kill_tasks_on_workers(self, task_ids: set[TaskId]) -> None:
        """Send KILL RPCs to workers for tasks that were running."""
        ...


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Args:
        state: Controller state containing jobs, tasks, and workers
        scheduler: Background scheduler for task dispatch (any object with wake() method)
        bundle_dir: Directory for storing uploaded bundles (optional)
    """

    def __init__(
        self,
        state: ControllerState,
        scheduler: SchedulerProtocol,
        bundle_dir: str | Path | None = None,
    ):
        self._state = state
        self._scheduler = scheduler
        self._bundle_dir = Path(bundle_dir) if bundle_dir else None

    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        """Submit a new job to the controller.

        The job is expanded into tasks based on the resources.replicas field
        (defaulting to 1). Each task has ID "{job_id}/task-{index}".
        """
        with rpc_error_handler("launching job"):
            if not request.name:
                raise ConnectError(Code.INVALID_ARGUMENT, "Job name is required")

            job_id = request.name

            if self._state.get_job(JobId(job_id)):
                raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists")

            # Handle bundle_blob: write to bundle_dir if provided
            if request.bundle_blob and self._bundle_dir:
                bundle_path = self._bundle_dir / job_id / "bundle.zip"
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                bundle_path.write_bytes(request.bundle_blob)

                request = cluster_pb2.Controller.LaunchJobRequest(
                    name=request.name,
                    serialized_entrypoint=request.serialized_entrypoint,
                    resources=request.resources,
                    environment=request.environment,
                    bundle_gcs_path=f"file://{bundle_path}",
                    bundle_hash=request.bundle_hash,
                    ports=list(request.ports),
                    scheduling_timeout_seconds=request.scheduling_timeout_seconds,
                    parent_job_id=request.parent_job_id,
                )

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

            task_statuses.append(
                cluster_pb2.TaskStatus(
                    task_id=str(task.task_id),
                    job_id=str(task.job_id),
                    task_index=task.task_index,
                    state=task.state,
                    worker_id=str(task.worker_id) if task.worker_id else "",
                    worker_address=worker_address,
                    started_at_ms=task.started_at_ms or 0,
                    finished_at_ms=task.finished_at_ms or 0,
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
                num_tasks=len(task_statuses),
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
            # Aggregate failure and preemption counts from all tasks
            tasks = self._state.get_job_tasks(j.job_id)
            total_failure_count = sum(task.failure_count for task in tasks)
            total_preemption_count = sum(task.preemption_count for task in tasks)

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

        return cluster_pb2.Controller.GetTaskStatusResponse(
            task=cluster_pb2.TaskStatus(
                task_id=str(task.task_id),
                job_id=str(task.job_id),
                task_index=task.task_index,
                state=task.state,
                worker_id=str(task.worker_id) if task.worker_id else "",
                worker_address=worker_address,
                started_at_ms=task.started_at_ms or 0,
                finished_at_ms=task.finished_at_ms or 0,
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

        task_statuses = []
        for task in tasks:
            # Look up worker address
            worker_address = ""
            if task.worker_id:
                worker = self._state.get_worker(task.worker_id)
                if worker:
                    worker_address = worker.address

            task_statuses.append(
                cluster_pb2.TaskStatus(
                    task_id=str(task.task_id),
                    job_id=str(task.job_id),
                    task_index=task.task_index,
                    state=task.state,
                    worker_id=str(task.worker_id) if task.worker_id else "",
                    worker_address=worker_address,
                    started_at_ms=task.started_at_ms or 0,
                    finished_at_ms=task.finished_at_ms or 0,
                    exit_code=task.exit_code or 0,
                    error=task.error or "",
                    current_attempt_id=task.current_attempt_id,
                )
            )

        return cluster_pb2.Controller.ListTasksResponse(tasks=task_statuses)

    def report_task_state(
        self,
        request: cluster_pb2.Controller.ReportTaskStateRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ReportTaskStateResponse:
        """Handle task state report from a worker.

        Workers report task state changes (BUILDING, RUNNING, SUCCEEDED, FAILED, etc.)
        via this RPC. The controller updates task state and aggregates to job state.
        """
        task_id = TaskId(request.task_id)
        task = self._state.get_task(task_id)
        if not task:
            logger.warning(f"Received task state report for unknown task {task_id}")
            return cluster_pb2.Controller.ReportTaskStateResponse()

        # Validate attempt_id matches current attempt
        if request.attempt_id != task.current_attempt_id:
            logger.warning(
                f"Received stale task state report: task_id={task_id} "
                f"expected_attempt={task.current_attempt_id} reported_attempt={request.attempt_id}"
            )
            return cluster_pb2.Controller.ReportTaskStateResponse()

        new_state = request.state

        txn = self._state.handle_event(
            TaskStateChangedEvent(
                task_id=task_id,
                new_state=new_state,
                error=request.error if request.error else None,
                exit_code=request.exit_code if request.exit_code else None,
            )
        )

        logger.debug(f"Task {task_id} reported state {new_state}")

        # Send kill RPCs to workers for any tasks that were killed as a result
        if txn.tasks_to_kill:
            self._scheduler.kill_tasks_on_workers(txn.tasks_to_kill)

        # Wake scheduler if task finished (may free capacity)
        if task.is_finished():
            self._scheduler.wake()

        return cluster_pb2.Controller.ReportTaskStateResponse()

    # --- Worker Management ---

    def register_worker(
        self,
        request: cluster_pb2.Controller.RegisterWorkerRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterWorkerResponse:
        """Register a new worker or process a heartbeat from an existing worker."""
        # Use WORKER_REGISTERED event for both new and existing workers
        self._state.handle_event(
            WorkerRegisteredEvent(
                worker_id=WorkerId(request.worker_id),
                address=request.address,
                metadata=request.metadata,
                timestamp_ms=now_ms(),
            )
        )
        self._scheduler.wake()
        return cluster_pb2.Controller.RegisterWorkerResponse(accepted=True)

    def list_workers(
        self,
        request: cluster_pb2.Controller.ListWorkersRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListWorkersResponse:
        """List all workers with their running task counts."""
        workers = [
            cluster_pb2.Controller.WorkerHealthStatus(
                worker_id=w.worker_id,
                healthy=w.healthy,
                consecutive_failures=w.consecutive_failures,
                last_heartbeat_ms=w.last_heartbeat_ms,
                running_job_ids=list(w.running_tasks),  # Now contains task IDs
            )
            for w in self._state.list_all_workers()
        ]
        return cluster_pb2.Controller.ListWorkersResponse(workers=workers)

    # --- Endpoint Management ---

    def register_endpoint(
        self,
        request: cluster_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterEndpointResponse:
        """Register a service endpoint.

        Endpoints are registered regardless of job state, but only become visible to clients
        (via lookup/list) when the job is RUNNING.
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
        """Look up a service endpoint by name. Only endpoints for RUNNING jobs are returned."""
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
        """List endpoints by name prefix. Only endpoints for RUNNING jobs are returned."""
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
