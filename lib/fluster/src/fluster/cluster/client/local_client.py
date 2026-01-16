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

"""Local in-process cluster client implementation.

This module provides LocalClusterClient, which implements the ClusterClient
protocol using in-process threads for job execution.
"""

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from fluster.cluster.client.job_info import JobInfo, set_job_info
from fluster.cluster.types import Entrypoint, is_job_finished
from fluster.rpc import cluster_pb2
from fluster.time_utils import ExponentialBackoff


@dataclass
class _LocalJob:
    """Internal job tracking state."""

    job_id: str
    future: Future
    state: int = cluster_pb2.JOB_STATE_PENDING
    error: str = ""
    result: Any = None
    started_at_ms: int = 0
    finished_at_ms: int = 0


@dataclass
class _LocalEndpoint:
    """Internal endpoint tracking state."""

    endpoint_id: str
    name: str
    address: str
    job_id: str
    metadata: dict[str, str]


class LocalClusterClient:
    """Cluster client for local/thread-based execution.

    This is a "dumb" implementation - all parameters are explicit, no context magic.
    Jobs run in threads with JobInfo contextvar injection.
    """

    def __init__(
        self,
        max_workers: int = 4,
        port_range: tuple[int, int] = (50000, 60000),
    ):
        """Initialize local cluster operations.

        Args:
            max_workers: Maximum concurrent job threads
            port_range: Port range for actor servers (inclusive start, exclusive end)
        """
        self._max_workers = max_workers
        self._port_range = port_range
        self._executor: ThreadPoolExecutor | None = None
        self._jobs: dict[str, _LocalJob] = {}
        self._endpoints: dict[str, _LocalEndpoint] = {}
        self._lock = threading.RLock()
        self._next_port = port_range[0]

    def start(self) -> None:
        """Start the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor.

        Args:
            wait: If True, wait for pending jobs to complete
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def _allocate_port(self) -> int:
        """Allocate a port from the configured range."""
        with self._lock:
            port = self._next_port
            self._next_port += 1
            if self._next_port >= self._port_range[1]:
                self._next_port = self._port_range[0]
        return port

    def submit_job(
        self,
        job_id: str,
        entrypoint: Entrypoint,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
    ) -> None:
        """Submit a job for local execution in a thread.

        Args:
            job_id: Full hierarchical job ID (e.g., "root/worker-0")
            entrypoint: Job entrypoint (callable + args/kwargs)
            resources: Resource requirements (ignored in local mode)
            environment: Environment configuration (ignored in local mode)
            ports: Port names to allocate (e.g., ["actor"])
            scheduling_timeout_seconds: Ignored in local mode (jobs start immediately)
        """
        del scheduling_timeout_seconds  # Unused in local execution
        if self._executor is None:
            raise RuntimeError("LocalClusterClient not started. Call start() first.")

        # Allocate requested ports
        allocated_ports = {port_name: self._allocate_port() for port_name in ports or []}

        # Create job info for this execution
        job_info = JobInfo(
            job_id=job_id,
            worker_id=f"local-worker-{threading.current_thread().ident}",
            ports=allocated_ports,
        )

        # Reject duplicates (must check under lock for thread safety)
        with self._lock:
            if job_id in self._jobs:
                raise ValueError(f"Job {job_id} already exists")

            # Create job tracking
            local_job = _LocalJob(
                job_id=job_id,
                future=Future(),
                state=cluster_pb2.JOB_STATE_PENDING,
                started_at_ms=int(time.time() * 1000),
            )
            self._jobs[job_id] = local_job

        # Submit to thread pool
        self._executor.submit(self._run_job, local_job, job_info, entrypoint)

    def _run_job(
        self,
        job: _LocalJob,
        job_info: JobInfo,
        entrypoint: Entrypoint,
    ) -> None:
        """Execute job entrypoint with job_info injection."""
        job.state = cluster_pb2.JOB_STATE_RUNNING

        try:
            # Inject job info into contextvar for this thread
            set_job_info(job_info)
            result = entrypoint.callable(*entrypoint.args, **entrypoint.kwargs)
            job.result = result
            job.state = cluster_pb2.JOB_STATE_SUCCEEDED
            job.future.set_result(result)
        except Exception as e:
            job.error = str(e)
            job.state = cluster_pb2.JOB_STATE_FAILED
            job.future.set_exception(e)
        finally:
            job.finished_at_ms = int(time.time() * 1000)

    def get_job_status(self, job_id: str) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Full job ID

        Returns:
            JobStatus proto with current state
        """
        with self._lock:
            job = self._jobs.get(job_id)

        if job is None:
            return cluster_pb2.JobStatus(
                job_id=job_id,
                state=cluster_pb2.JOB_STATE_UNSCHEDULABLE,  # type: ignore[arg-type]
            )

        return cluster_pb2.JobStatus(
            job_id=job_id,
            state=job.state,  # type: ignore[arg-type]
            error=job.error,
            started_at_ms=job.started_at_ms,
            finished_at_ms=job.finished_at_ms,
        )

    def wait_for_job(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete with exponential backoff polling.

        Args:
            job_id: Full job ID
            timeout: Maximum time to wait in seconds
            poll_interval: Maximum time between status checks

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        start = time.monotonic()
        backoff = ExponentialBackoff(initial=0.1, maximum=poll_interval)

        while True:
            status = self.get_job_status(job_id)
            if is_job_finished(status.state):
                return status

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            interval = backoff.next_interval()
            remaining = timeout - elapsed
            time.sleep(min(interval, remaining))

    def terminate_job(self, job_id: str) -> None:
        """Terminate a running job.

        Note: In local mode, jobs cannot be forcefully terminated.
        This marks the job as killed but the thread continues until completion.

        Args:
            job_id: Full job ID
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job and not is_job_finished(job.state):
                job.state = cluster_pb2.JOB_STATE_KILLED
                job.error = "Terminated by user"
                job.finished_at_ms = int(time.time() * 1000)

    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint in local storage.

        Args:
            name: Full endpoint name (with namespace prefix if needed)
            address: Address where actor is listening (host:port)
            job_id: Job ID that owns this endpoint
            metadata: Optional metadata

        Returns:
            Endpoint ID (unique local identifier)
        """
        endpoint_id = f"local-ep-{uuid.uuid4().hex[:8]}"
        endpoint = _LocalEndpoint(
            endpoint_id=endpoint_id,
            name=name,
            address=address,
            job_id=job_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._endpoints[endpoint_id] = endpoint

        return endpoint_id

    def unregister_endpoint(self, endpoint_id: str) -> None:
        """Unregister an endpoint from local storage.

        Args:
            endpoint_id: Endpoint ID to remove
        """
        with self._lock:
            self._endpoints.pop(endpoint_id, None)

    def list_endpoints(self, prefix: str) -> list[cluster_pb2.Controller.Endpoint]:
        """List endpoints matching a prefix.

        Args:
            prefix: Name prefix to match (e.g., "abc123/")

        Returns:
            List of matching endpoints
        """
        with self._lock:
            matches = [
                cluster_pb2.Controller.Endpoint(
                    endpoint_id=ep.endpoint_id,
                    name=ep.name,
                    address=ep.address,
                    job_id=ep.job_id,
                    metadata=ep.metadata,
                )
                for ep in self._endpoints.values()
                if ep.name.startswith(prefix)
            ]
        return matches
