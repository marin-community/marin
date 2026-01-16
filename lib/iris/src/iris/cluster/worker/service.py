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

"""WorkerService RPC implementation using Connect RPC."""

import re
import time
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.worker.worker_types import Job
from iris.rpc import cluster_pb2
from iris.rpc.errors import rpc_error_handler


class JobProvider(Protocol):
    """Protocol for job management operations."""

    def submit_job(self, request: cluster_pb2.Worker.RunJobRequest) -> str: ...
    def get_job(self, job_id: str) -> Job | None: ...
    def list_jobs(self) -> list[Job]: ...
    def kill_job(self, job_id: str, term_timeout_ms: int = 5000) -> bool: ...
    def get_logs(self, job_id: str, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]: ...


class WorkerServiceImpl:
    """Implementation of WorkerService RPC interface."""

    def __init__(self, provider: JobProvider):
        self._provider = provider
        self._start_time = time.time()

    def run_job(
        self,
        request: cluster_pb2.Worker.RunJobRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.RunJobResponse:
        with rpc_error_handler("running job"):
            job_id = self._provider.submit_job(request)
            job = self._provider.get_job(job_id)

            if not job:
                raise ConnectError(Code.INTERNAL, f"Job {job_id} not found after submission")

            return cluster_pb2.Worker.RunJobResponse(
                job_id=job_id,
                state=job.to_proto().state,
            )

    def get_job_status(
        self,
        request: cluster_pb2.Worker.GetJobStatusRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.JobStatus:
        job = self._provider.get_job(request.job_id)
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        status = job.to_proto()
        if request.include_result and job.result:
            status.serialized_result = job.result
        return status

    def list_jobs(
        self,
        _request: cluster_pb2.Worker.ListJobsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.ListJobsResponse:
        jobs = self._provider.list_jobs()
        return cluster_pb2.Worker.ListJobsResponse(
            jobs=[job.to_proto() for job in jobs],
        )

    def fetch_logs(
        self,
        request: cluster_pb2.Worker.FetchLogsRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Worker.FetchLogsResponse:
        start_line = request.filter.start_line if request.filter.start_line else 0
        logs = self._provider.get_logs(request.job_id, start_line=start_line)

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

        return cluster_pb2.Worker.FetchLogsResponse(logs=result)

    def kill_job(
        self,
        request: cluster_pb2.Worker.KillJobRequest,
        _ctx: RequestContext,
    ) -> cluster_pb2.Empty:
        job = self._provider.get_job(request.job_id)
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        success = self._provider.kill_job(
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
    ) -> cluster_pb2.Worker.HealthResponse:
        jobs = self._provider.list_jobs()
        running = sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_RUNNING)

        return cluster_pb2.Worker.HealthResponse(
            healthy=True,
            uptime_ms=int((time.time() - self._start_time) * 1000),
            running_jobs=running,
        )
