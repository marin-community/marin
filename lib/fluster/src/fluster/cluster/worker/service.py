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

"""WorkerService RPC implementation using Connect RPC.

Implements the WorkerService protocol defined in cluster.proto.
Provides job execution, status, logs, and health monitoring endpoints.
"""

import re
import time
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from fluster import cluster_pb2
from fluster.cluster.worker.manager import JobManager


class WorkerServiceImpl:
    """Implementation of WorkerService RPC interface.

    Provides endpoints for:
    - run_job: Submit job for execution
    - get_job_status: Query job status
    - list_jobs: List jobs (optionally filtered)
    - fetch_logs: Get logs with filtering
    - kill_job: Terminate job
    - health_check: Worker health status
    """

    def __init__(self, manager: JobManager):
        self._manager = manager
        self._start_time = time.time()

    def run_job(
        self,
        request: cluster_pb2.RunJobRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.RunJobResponse:
        """Submit job for execution."""
        job_id = self._manager.submit_job(request)
        job = self._manager.get_job(job_id)

        if not job:
            raise ConnectError(Code.INTERNAL, f"Job {job_id} not found after submission")

        return cluster_pb2.RunJobResponse(
            job_id=job_id,
            state=job.to_proto().state,
        )

    def get_job_status(
        self,
        request: cluster_pb2.GetStatusRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.JobStatus:
        """Get job status."""
        job = self._manager.get_job(request.job_id)
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")
        return job.to_proto()

    def list_jobs(
        self,
        _request: cluster_pb2.ListJobsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.ListJobsResponse:
        """List jobs.

        Note: namespace filtering is not implemented in this stage as jobs
        are stored without namespace information. Empty string is treated
        as "list all jobs".
        """
        jobs = self._manager.list_jobs()
        return cluster_pb2.ListJobsResponse(
            jobs=[job.to_proto() for job in jobs],
        )

    def fetch_logs(
        self,
        request: cluster_pb2.FetchLogsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.FetchLogsResponse:
        """Fetch job logs with filtering.

        Supports:
        - start_line: Line offset. Negative values for tailing (e.g., -100 for last 100 lines)
        - start_ms/end_ms: Time range filter
        - regex: Content filter
        - max_lines: Limit results
        """
        # Get logs with start_line handling (negative = tail)
        start_line = request.filter.start_line if request.filter.start_line else 0
        logs = self._manager.get_logs(request.job_id, start_line=start_line)

        # Apply additional filters
        result = []
        for entry in logs:
            # Time range filter (start_ms is exclusive for incremental polling)
            if request.filter.start_ms and entry.timestamp_ms <= request.filter.start_ms:
                continue
            if request.filter.end_ms and entry.timestamp_ms > request.filter.end_ms:
                continue
            # TODO: Regex filter is vulnerable to DoS via catastrophic backtracking.
            # Malicious regex like (a+)+ can cause minutes of CPU time. Consider using
            # the re2 library or adding timeout/complexity limits.
            # Regex filter
            if request.filter.regex:
                if not re.search(request.filter.regex, entry.data):
                    continue

            result.append(entry)

            # Max lines limit
            if request.filter.max_lines and len(result) >= request.filter.max_lines:
                break

        return cluster_pb2.FetchLogsResponse(logs=result)

    def kill_job(
        self,
        request: cluster_pb2.KillJobRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Empty:
        """Kill running job."""
        # Check if job exists first
        job = self._manager.get_job(request.job_id)
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        success = self._manager.kill_job(
            request.job_id,
            term_timeout_ms=request.term_timeout_ms or 5000,
        )
        if not success:
            # Job exists but is already in terminal state
            state_name = cluster_pb2.JobState.Name(job.status)
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Job {request.job_id} already completed with state {state_name}",
            )
        return cluster_pb2.Empty()

    def health_check(
        self,
        _request: cluster_pb2.Empty,
        _ctx: RequestContext,
    ) -> cluster_pb2.HealthResponse:
        """Worker health status."""
        jobs = self._manager.list_jobs()
        running = sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_RUNNING)

        return cluster_pb2.HealthResponse(
            healthy=True,
            uptime_ms=int((time.time() - self._start_time) * 1000),
            running_jobs=running,
        )
