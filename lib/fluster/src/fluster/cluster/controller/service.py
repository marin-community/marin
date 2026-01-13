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
- Worker heartbeat (heartbeat)
- Worker listing (list_workers)

The service layer is thin, delegating most logic to the ControllerState and
Scheduler. It focuses on proto message conversion and error handling.
"""

import time
import uuid
from pathlib import Path
from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from fluster import cluster_pb2
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import JobId, WorkerId, is_job_finished


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Provides HTTP handlers for job management operations. Each method accepts
    a protobuf request message and returns a protobuf response message.

    Args:
        state: Controller state containing jobs and workers
        scheduler: Background scheduler for job dispatch
        bundle_dir: Directory for storing uploaded bundles (optional)
    """

    def __init__(
        self,
        state: ControllerState,
        scheduler: Scheduler,
        bundle_dir: str | Path | None = None,
    ):
        self._state = state
        self._scheduler = scheduler
        self._bundle_dir = Path(bundle_dir) if bundle_dir else None

    def launch_job(
        self,
        request: cluster_pb2.LaunchJobRequest,
        ctx: Any,
    ) -> cluster_pb2.LaunchJobResponse:
        """Submit a new job to the controller.

        Creates a new job with a unique ID, adds it to the controller state,
        and wakes the scheduler to attempt immediate dispatch.

        If bundle_blob is provided and bundle_dir is configured, writes the
        bundle to disk and updates bundle_gcs_path to a file:// URL.

        Args:
            request: Job launch request with entrypoint and resource spec
            ctx: Request context (unused in v0)

        Returns:
            LaunchJobResponse containing the assigned job_id
        """
        job_id = str(uuid.uuid4())

        # Handle bundle_blob: write to bundle_dir if provided
        if request.bundle_blob and self._bundle_dir:
            bundle_path = self._bundle_dir / job_id / "bundle.zip"
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            bundle_path.write_bytes(request.bundle_blob)

            # Update the request with file:// path
            request = cluster_pb2.LaunchJobRequest(
                name=request.name,
                serialized_entrypoint=request.serialized_entrypoint,
                resources=request.resources,
                environment=request.environment,
                bundle_gcs_path=f"file://{bundle_path}",
                bundle_hash=request.bundle_hash,
            )

        job = ControllerJob(
            job_id=JobId(job_id),
            request=request,
            submitted_at_ms=int(time.time() * 1000),
        )

        self._state.add_job(job)
        self._state.log_action("job_submitted", job_id=job.job_id, details=request.name)
        self._scheduler.wake()  # Try to schedule immediately

        return cluster_pb2.LaunchJobResponse(job_id=job_id)

    def get_job_status(
        self,
        request: cluster_pb2.GetJobStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.GetJobStatusResponse:
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
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Terminate a running job.

        Marks the job as KILLED in the controller state. Note that in v0,
        this does not send an actual kill signal to the worker - that is
        deferred to a future implementation.

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

        # Idempotent: if already in terminal state, do nothing
        if is_job_finished(job.state):
            return cluster_pb2.Empty()

        # TODO: Send kill to worker
        job.state = cluster_pb2.JOB_STATE_KILLED
        job.finished_at_ms = int(time.time() * 1000)
        self._state.log_action("job_killed", job_id=job.job_id)

        return cluster_pb2.Empty()

    def list_jobs(
        self,
        request: cluster_pb2.ListJobsRequest,
        ctx: Any,
    ) -> cluster_pb2.ListJobsResponse:
        """List all jobs.

        Returns a list of all jobs in the controller, regardless of state.
        Note that the namespace field in the request is ignored in v0.

        Args:
            request: List request (namespace field currently ignored)
            ctx: Request context (unused in v0)

        Returns:
            ListJobsResponse containing all jobs as JobStatus protos
        """
        jobs = [
            cluster_pb2.JobStatus(
                job_id=j.job_id,
                state=j.state,
                worker_id=j.worker_id or "",
                error=j.error or "",
                exit_code=j.exit_code or 0,
                started_at_ms=j.started_at_ms or 0,
                finished_at_ms=j.finished_at_ms or 0,
            )
            for j in self._state.list_all_jobs()
        ]
        return cluster_pb2.ListJobsResponse(jobs=jobs)

    def register_worker(
        self,
        request: cluster_pb2.RegisterWorkerRequest,
        ctx: Any,
    ) -> cluster_pb2.RegisterWorkerResponse:
        """Register a new worker with the controller.

        Workers register themselves on startup and provide their address and
        resource capabilities. The controller adds them to the worker registry
        and wakes the scheduler to potentially dispatch pending jobs.

        Args:
            request: Worker registration request
            ctx: Request context (unused in v0)

        Returns:
            RegisterWorkerResponse with acceptance status
        """
        worker = ControllerWorker(
            worker_id=WorkerId(request.worker_id),
            address=request.address,
            resources=request.resources,
            last_heartbeat_ms=int(time.time() * 1000),
        )
        self._state.add_worker(worker)
        self._state.log_action(
            "worker_registered",
            worker_id=worker.worker_id,
            details=f"address={request.address}",
        )
        self._scheduler.wake()  # Try to schedule jobs on new worker

        return cluster_pb2.RegisterWorkerResponse(accepted=True)

    def heartbeat(
        self,
        request: cluster_pb2.HeartbeatRequest,
        ctx: Any,
    ) -> cluster_pb2.HeartbeatResponse:
        """Process worker heartbeat.

        Workers send periodic heartbeats to indicate they are alive. The controller
        updates the worker's last heartbeat timestamp and returns the status of
        jobs that have been modified since the worker's last check.

        Args:
            request: Heartbeat request with worker_id and since_ms
            ctx: Request context (unused in v0)

        Returns:
            HeartbeatResponse with job statuses

        Raises:
            ConnectError: If worker is not found (Code.NOT_FOUND)
        """
        worker = self._state.get_worker(WorkerId(request.worker_id))
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {request.worker_id} not found")

        now_ms = int(time.time() * 1000)
        worker.last_heartbeat_ms = now_ms
        worker.consecutive_failures = 0

        # Return jobs assigned to this worker
        jobs = []
        for job_id in list(worker.running_jobs):
            job = self._state.get_job(job_id)
            if job:
                jobs.append(
                    cluster_pb2.JobStatus(
                        job_id=job.job_id,
                        state=job.state,
                        error=job.error or "",
                        exit_code=job.exit_code or 0,
                        started_at_ms=job.started_at_ms or 0,
                        finished_at_ms=job.finished_at_ms or 0,
                        worker_id=job.worker_id or "",
                    )
                )

        return cluster_pb2.HeartbeatResponse(jobs=jobs, timestamp_ms=now_ms)

    def list_workers(
        self,
        request: cluster_pb2.ListWorkersRequest,
        ctx: Any,
    ) -> cluster_pb2.ListWorkersResponse:
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
            cluster_pb2.WorkerHealthStatus(
                worker_id=w.worker_id,
                healthy=w.healthy,
                consecutive_failures=w.consecutive_failures,
                last_heartbeat_ms=w.last_heartbeat_ms,
                running_job_ids=list(w.running_jobs),
            )
            for w in self._state.list_all_workers()
        ]
        return cluster_pb2.ListWorkersResponse(workers=workers)

    # Endpoint registry methods - not implemented in v0

    def register_endpoint(
        self,
        request: cluster_pb2.RegisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.RegisterEndpointResponse:
        """Register a service endpoint. Not implemented in v0."""
        raise ConnectError(Code.UNIMPLEMENTED, "Endpoint registry not implemented in v0")

    def unregister_endpoint(
        self,
        request: cluster_pb2.UnregisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Unregister a service endpoint. Not implemented in v0."""
        raise ConnectError(Code.UNIMPLEMENTED, "Endpoint registry not implemented in v0")

    def lookup_endpoint(
        self,
        request: cluster_pb2.LookupEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.LookupEndpointResponse:
        """Look up a service endpoint. Not implemented in v0."""
        raise ConnectError(Code.UNIMPLEMENTED, "Endpoint registry not implemented in v0")

    def list_endpoints(
        self,
        request: cluster_pb2.ListEndpointsRequest,
        ctx: Any,
    ) -> cluster_pb2.ListEndpointsResponse:
        """List service endpoints. Not implemented in v0."""
        raise ConnectError(Code.UNIMPLEMENTED, "Endpoint registry not implemented in v0")
