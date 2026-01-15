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

"""Controller RPC service implementation.

This module provides the ControllerServiceImpl class, which implements the
RPC handlers for the ControllerService. It handles:
- Job submission (launch_job)
- Job status queries (get_job_status)
- Job termination (terminate_job)
- Job listing (list_jobs)
- Worker registration (register_worker)
- Worker listing (list_workers)
- Endpoint registry operations (register/unregister/lookup/list endpoints)

The service layer is thin, delegating most logic to the ControllerState and
Scheduler. It focuses on proto message conversion and error handling.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from connectrpc.code import Code

from connectrpc.errors import ConnectError

from fluster.cluster.controller.state import ControllerEndpoint, ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import JobId, WorkerId, is_job_finished
from fluster.rpc import cluster_pb2
from fluster.rpc.errors import rpc_error_handler

logger = logging.getLogger(__name__)


class SchedulerProtocol(Protocol):
    """Protocol for scheduler operations used by ControllerServiceImpl."""

    def wake(self) -> None:
        """Signal scheduler to run immediately."""
        ...


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Provides HTTP handlers for job management operations. Each method accepts
    a protobuf request message and returns a protobuf response message.

    Args:
        state: Controller state containing jobs and workers
        scheduler: Background scheduler for job dispatch (any object with wake() method)
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

        Creates a new job with a unique ID, adds it to the controller state,
        and wakes the scheduler to attempt immediate dispatch.

        If bundle_blob is provided and bundle_dir is configured, writes the
        bundle to disk and updates bundle_gcs_path to a file:// URL.

        If parent_job_id is provided in the request, the job is tracked as a
        child of that parent job. When the parent is terminated, all children
        will be terminated as well.

        Args:
            request: Job launch request with entrypoint and resource spec
            ctx: Request context (unused in v0)

        Returns:
            LaunchJobResponse containing the assigned job_id
        """
        with rpc_error_handler("launching job"):
            job_id = str(uuid.uuid4())

            # Handle bundle_blob: write to bundle_dir if provided
            if request.bundle_blob and self._bundle_dir:
                bundle_path = self._bundle_dir / job_id / "bundle.zip"
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                bundle_path.write_bytes(request.bundle_blob)

                # Update the request with file:// path
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

            # Resolve parent_job_id: empty string means no parent
            parent_job_id = JobId(request.parent_job_id) if request.parent_job_id else None

            job = ControllerJob(
                job_id=JobId(job_id),
                request=request,
                submitted_at_ms=int(time.time() * 1000),
                parent_job_id=parent_job_id,
            )

            self._state.add_job(job)
            self._state.log_action("job_submitted", job_id=job.job_id, details=request.name)
            self._scheduler.wake()  # Try to schedule immediately

            return cluster_pb2.Controller.LaunchJobResponse(job_id=job_id)

    def get_job_status(
        self,
        request: cluster_pb2.Controller.GetJobStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        """Get status of a specific job.

        Args:
            request: Request containing job_id
            ctx: Request context (unused in v0)

        Returns:
            GetJobStatusResponse with JobStatus proto

        Raises:
            ConnectError: If job is not found (Code.NOT_FOUND)
        """
        job = self._state.get_job(JobId(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        worker_address = ""
        if job.worker_id:
            worker = self._state.get_worker(job.worker_id)
            if worker:
                worker_address = worker.address

        return cluster_pb2.Controller.GetJobStatusResponse(
            job=cluster_pb2.JobStatus(
                job_id=job.job_id,
                state=job.state,
                error=job.error or "",
                exit_code=job.exit_code or 0,
                started_at_ms=job.started_at_ms or 0,
                finished_at_ms=job.finished_at_ms or 0,
                worker_id=job.worker_id or "",
                worker_address=worker_address,
                parent_job_id=job.parent_job_id or "",
            )
        )

    def terminate_job(
        self,
        request: cluster_pb2.Controller.TerminateJobRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Terminate a running job and all its children.

        Marks the job as KILLED in the controller state. Cascade termination
        is performed depth-first: all children are terminated before the parent.
        Note that in v0, this does not send an actual kill signal to the worker -
        that is deferred to a future implementation.

        Args:
            request: Request containing job_id
            ctx: Request context (unused in v0)

        Returns:
            Empty response

        Raises:
            ConnectError: If job is not found (Code.NOT_FOUND)
        """
        job = self._state.get_job(JobId(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        # Cascade terminate children first (depth-first)
        self._terminate_job_tree(job)

        return cluster_pb2.Empty()

    def _terminate_job_tree(self, job: ControllerJob) -> None:
        """Recursively terminate a job and all its descendants (depth-first).

        Children are terminated before the parent to ensure proper cascade.
        """
        # First, terminate all children recursively
        children = self._state.get_children(job.job_id)
        for child in children:
            self._terminate_job_tree(child)

        # Now terminate this job if not already in terminal state
        if is_job_finished(job.state):
            return

        # TODO: Send kill to worker
        job.state = cluster_pb2.JOB_STATE_KILLED
        job.finished_at_ms = int(time.time() * 1000)
        self._state.log_action("job_killed", job_id=job.job_id)

    def list_jobs(
        self,
        request: cluster_pb2.Controller.ListJobsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListJobsResponse:
        """List all jobs.

        Returns a list of all jobs in the controller, regardless of state.
        Note that the namespace field in the request is ignored in v0.

        Args:
            request: List request (namespace field currently ignored)
            ctx: Request context (unused in v0)

        Returns:
            ListJobsResponse containing all jobs as JobStatus protos
        """
        jobs = []
        for j in self._state.list_all_jobs():
            worker_address = ""
            if j.worker_id:
                worker = self._state.get_worker(j.worker_id)
                if worker:
                    worker_address = worker.address
            jobs.append(
                cluster_pb2.JobStatus(
                    job_id=j.job_id,
                    state=j.state,
                    worker_id=j.worker_id or "",
                    worker_address=worker_address,
                    error=j.error or "",
                    exit_code=j.exit_code or 0,
                    started_at_ms=j.started_at_ms or 0,
                    finished_at_ms=j.finished_at_ms or 0,
                    parent_job_id=j.parent_job_id or "",
                )
            )
        return cluster_pb2.Controller.ListJobsResponse(jobs=jobs)

    def register_worker(
        self,
        request: cluster_pb2.Controller.RegisterWorkerRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterWorkerResponse:
        """Register a new worker or process a heartbeat from an existing worker.

        Workers send periodic heartbeats via this endpoint. If the worker already
        exists, we update its heartbeat timestamp and metadata. If it's a new
        worker, we add it to the registry and wake the scheduler.

        Args:
            request: Worker registration request (also used as heartbeat)
            ctx: Request context (unused in v0)

        Returns:
            RegisterWorkerResponse with acceptance status
        """
        now_ms = int(time.time() * 1000)
        worker_id = WorkerId(request.worker_id)

        # Check if worker already exists (heartbeat case)
        existing = self._state.get_worker(worker_id)
        if existing:
            # Update heartbeat timestamp and worker info
            existing.last_heartbeat_ms = now_ms
            existing.resources = request.resources
            existing.metadata = request.metadata
            existing.healthy = True  # Worker is alive
            return cluster_pb2.Controller.RegisterWorkerResponse(accepted=True)

        # New worker registration
        worker = ControllerWorker(
            worker_id=worker_id,
            address=request.address,
            resources=request.resources,
            metadata=request.metadata,
            last_heartbeat_ms=now_ms,
        )
        self._state.add_worker(worker)
        self._state.log_action(
            "worker_registered",
            worker_id=worker.worker_id,
            details=f"address={request.address}",
        )
        self._scheduler.wake()  # Try to schedule jobs on new worker

        return cluster_pb2.Controller.RegisterWorkerResponse(accepted=True)

    def list_workers(
        self,
        request: cluster_pb2.Controller.ListWorkersRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListWorkersResponse:
        """List all registered workers.

        Returns health status for all workers in the controller, including
        healthy and unhealthy workers.

        Args:
            request: List workers request (currently ignored)
            ctx: Request context (unused in v0)

        Returns:
            ListWorkersResponse with worker health statuses
        """
        workers = [
            cluster_pb2.Controller.WorkerHealthStatus(
                worker_id=w.worker_id,
                healthy=w.healthy,
                consecutive_failures=w.consecutive_failures,
                last_heartbeat_ms=w.last_heartbeat_ms,
                running_job_ids=list(w.running_jobs),
            )
            for w in self._state.list_all_workers()
        ]
        return cluster_pb2.Controller.ListWorkersResponse(workers=workers)

    def report_job_state(
        self,
        request: cluster_pb2.Controller.ReportJobStateRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ReportJobStateResponse:
        """Report job state change from worker.

        Workers call this when jobs transition to terminal states (SUCCEEDED, FAILED, KILLED).
        The controller updates its job state and removes the job from the worker's running set.

        Args:
            request: Job state report from worker
            ctx: Request context (unused in v0)

        Returns:
            ReportJobStateResponse (empty)
        """
        job_id = JobId(request.job_id)
        worker_id = WorkerId(request.worker_id)

        job = self._state.get_job(job_id)
        if not job:
            logger.warning("Received job state report for unknown job %s from worker %s", job_id, worker_id)
            return cluster_pb2.Controller.ReportJobStateResponse()

        # Verify this report is from the assigned worker
        if job.worker_id != worker_id:
            logger.warning(
                "Received job state report from wrong worker: job_id=%s expected_worker=%s reporting_worker=%s",
                job_id,
                job.worker_id,
                worker_id,
            )
            return cluster_pb2.Controller.ReportJobStateResponse()

        # Update job state
        job.state = request.state
        job.exit_code = request.exit_code
        job.error = request.error or None
        job.finished_at_ms = request.finished_at_ms

        # Remove from worker's running jobs
        worker = self._state.get_worker(worker_id)
        if worker:
            worker.running_jobs.discard(job_id)

        # Clean up endpoints for finished job
        if is_job_finished(job.state):
            self._state.remove_endpoints_for_job(job_id)

        self._state.log_action(
            "job_completed",
            job_id=job_id,
            worker_id=worker_id,
            details=f"state={request.state}, exit_code={request.exit_code}",
        )

        return cluster_pb2.Controller.ReportJobStateResponse()

    # Endpoint registry methods

    def register_endpoint(
        self,
        request: cluster_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterEndpointResponse:
        """Register a service endpoint.

        Validates that the job exists and is RUNNING before registering.
        Endpoints are automatically removed when jobs terminate.

        Args:
            request: Endpoint registration request
            ctx: Request context (unused)

        Returns:
            RegisterEndpointResponse with assigned endpoint_id

        Raises:
            ConnectError: If job is not found or not running
        """
        with rpc_error_handler("registering endpoint"):
            endpoint_id = str(uuid.uuid4())

            # Validate job exists and is running
            job = self._state.get_job(JobId(request.job_id))
            if not job:
                raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")
            if job.state != cluster_pb2.JOB_STATE_RUNNING:
                raise ConnectError(Code.FAILED_PRECONDITION, f"Job {request.job_id} is not running")

            endpoint = ControllerEndpoint(
                endpoint_id=endpoint_id,
                name=request.name,  # Expected to be prefixed: "{root_job_id}/{actor_name}"
                address=request.address,
                job_id=JobId(request.job_id),
                metadata=dict(request.metadata),
                registered_at_ms=int(time.time() * 1000),
            )
            self._state.add_endpoint(endpoint)
            self._state.log_action(
                "endpoint_registered",
                job_id=job.job_id,
                details=f"{request.name} at {request.address}",
            )
            return cluster_pb2.Controller.RegisterEndpointResponse(endpoint_id=endpoint_id)

    def unregister_endpoint(
        self,
        request: cluster_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Unregister a service endpoint.

        Removes an endpoint from the registry. This is idempotent - no error
        if the endpoint doesn't exist.

        Args:
            request: Endpoint unregistration request
            ctx: Request context (unused)

        Returns:
            Empty response
        """
        endpoint = self._state.remove_endpoint(request.endpoint_id)
        if endpoint:
            self._state.log_action(
                "endpoint_unregistered",
                job_id=endpoint.job_id,
                details=endpoint.name,
            )
        return cluster_pb2.Empty()

    def lookup_endpoint(
        self,
        request: cluster_pb2.Controller.LookupEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.LookupEndpointResponse:
        """Look up a service endpoint by name.

        Returns the first endpoint matching the full prefixed name.
        Only endpoints for RUNNING jobs are returned.

        Args:
            request: Lookup request with full prefixed name (e.g., "abc123/my-actor")
            ctx: Request context (unused)

        Returns:
            LookupEndpointResponse with first matching endpoint (empty if not found)
        """
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
        """List endpoints by name prefix.

        Returns all endpoints matching the prefix.
        Only endpoints for RUNNING jobs are returned.

        Prefix is expected to include the namespace (e.g., "abc123/" or "abc123/my-actor").

        Args:
            request: List request with prefix (includes namespace)
            ctx: Request context (unused)

        Returns:
            ListEndpointsResponse with matching endpoints
        """
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
