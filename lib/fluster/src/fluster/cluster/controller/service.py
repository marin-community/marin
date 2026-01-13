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

The service layer is thin, delegating most logic to the ControllerState and
Scheduler. It focuses on proto message conversion and error handling.
"""

import time
import uuid
from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from fluster import cluster_pb2
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.state import ControllerJob, ControllerState
from fluster.cluster.types import JobId, is_job_finished


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Provides HTTP handlers for job management operations. Each method accepts
    a protobuf request message and returns a protobuf response message.

    Args:
        state: Controller state containing jobs and workers
        scheduler: Background scheduler for job dispatch
    """

    def __init__(self, state: ControllerState, scheduler: Scheduler):
        self._state = state
        self._scheduler = scheduler

    def launch_job(
        self,
        request: cluster_pb2.LaunchJobRequest,
        _ctx: Any,
    ) -> cluster_pb2.LaunchJobResponse:
        """Submit a new job to the controller.

        Creates a new job with a unique ID, adds it to the controller state,
        and wakes the scheduler to attempt immediate dispatch.

        Args:
            request: Job launch request with entrypoint and resource spec
            _ctx: Request context (unused in v0)

        Returns:
            LaunchJobResponse containing the assigned job_id
        """
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
        _ctx: Any,
    ) -> cluster_pb2.GetJobStatusResponse:
        """Get status of a specific job.

        Args:
            request: Request containing job_id
            _ctx: Request context (unused in v0)

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
        _ctx: Any,
    ) -> cluster_pb2.Empty:
        """Terminate a running job.

        Marks the job as KILLED in the controller state. Note that in v0,
        this does not send an actual kill signal to the worker - that is
        deferred to a future implementation.

        Args:
            request: Request containing job_id
            _ctx: Request context (unused in v0)

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

        return cluster_pb2.Empty()

    def list_jobs(
        self,
        request: cluster_pb2.ListJobsRequest,
        _ctx: Any,
    ) -> cluster_pb2.ListJobsResponse:
        """List all jobs.

        Returns a list of all jobs in the controller, regardless of state.
        Note that the namespace field in the request is ignored in v0.

        Args:
            request: List request (namespace field currently ignored)
            _ctx: Request context (unused in v0)

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
