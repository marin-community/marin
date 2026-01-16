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

"""RPC-based cluster client implementation."""

import time

import cloudpickle

from iris.cluster.types import Entrypoint, is_job_finished
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync, WorkerServiceClientSync
from iris.time_utils import ExponentialBackoff


class RemoteClusterClient:
    """Cluster client via RPC to controller.

    All parameters are explicit, no context magic. Takes full job IDs, full endpoint names, etc.
    """

    def __init__(
        self,
        controller_address: str,
        bundle_gcs_path: str | None = None,
        bundle_blob: bytes | None = None,
        timeout_ms: int = 30000,
    ):
        """Initialize RPC cluster operations.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            bundle_gcs_path: GCS path to workspace bundle for job inheritance
            bundle_blob: Workspace bundle as bytes (for initial job submission)
            timeout_ms: RPC timeout in milliseconds
        """
        self._address = controller_address
        self._bundle_gcs_path = bundle_gcs_path
        self._bundle_blob = bundle_blob
        self._timeout_ms = timeout_ms
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
        )
        self._worker_clients: dict[str, WorkerServiceClientSync] = {}

    def _get_worker_client(self, worker_address: str) -> WorkerServiceClientSync:
        if worker_address not in self._worker_clients:
            self._worker_clients[worker_address] = WorkerServiceClientSync(
                address=f"http://{worker_address}",
                timeout_ms=self._timeout_ms,
            )
        return self._worker_clients[worker_address]

    def submit_job(
        self,
        job_id: str,
        entrypoint: Entrypoint,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
    ) -> None:
        serialized = cloudpickle.dumps(entrypoint)

        env_config = cluster_pb2.EnvironmentConfig(
            workspace=environment.workspace if environment else "/app",
            pip_packages=list(environment.pip_packages) if environment else [],
            env_vars=dict(environment.env_vars) if environment else {},
            extras=list(environment.extras) if environment else [],
        )

        # Determine parent job ID (all but last component)
        parts = job_id.rsplit("/", 1)
        parent_job_id = parts[0] if len(parts) > 1 else ""

        # Use bundle_gcs_path if available, otherwise use bundle_blob
        if self._bundle_gcs_path:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=job_id,
                serialized_entrypoint=serialized,
                resources=resources,
                environment=env_config,
                bundle_gcs_path=self._bundle_gcs_path,
                ports=ports or [],
                parent_job_id=parent_job_id,
                scheduling_timeout_seconds=scheduling_timeout_seconds,
            )
        else:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=job_id,
                serialized_entrypoint=serialized,
                resources=resources,
                environment=env_config,
                bundle_blob=self._bundle_blob or b"",
                ports=ports or [],
                parent_job_id=parent_job_id,
                scheduling_timeout_seconds=scheduling_timeout_seconds,
            )
        self._client.launch_job(request)

    def get_job_status(self, job_id: str) -> cluster_pb2.JobStatus:
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self._client.get_job_status(request)
        return response.job

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
            job_info = self.get_job_status(job_id)
            if is_job_finished(job_info.state):
                return job_info

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            interval = backoff.next_interval()
            remaining = timeout - elapsed
            time.sleep(min(interval, remaining))

    def terminate_job(self, job_id: str) -> None:
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self._client.terminate_job(request)

    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=name,
            address=address,
            job_id=job_id,
            metadata=metadata or {},
        )
        response = self._client.register_endpoint(request)
        return response.endpoint_id

    def unregister_endpoint(self, endpoint_id: str) -> None:
        """Unregister an endpoint via RPC.

        This is a no-op for the RPC implementation. The controller automatically
        cleans up endpoints when jobs terminate, so explicit unregistration
        is not required.

        Args:
            endpoint_id: Endpoint ID (ignored)
        """
        # No-op: controller auto-cleans endpoints on job termination
        del endpoint_id

    def list_endpoints(self, prefix: str) -> list[cluster_pb2.Controller.Endpoint]:
        request = cluster_pb2.Controller.ListEndpointsRequest(prefix=prefix)
        response = self._client.list_endpoints(request)
        return list(response.endpoints)

    def list_jobs(self) -> list[cluster_pb2.JobStatus]:
        request = cluster_pb2.Controller.ListJobsRequest()
        response = self._client.list_jobs(request)
        return list(response.jobs)

    def shutdown(self, wait: bool = True) -> None:
        del wait
        for client in self._worker_clients.values():
            client.close()
        self._worker_clients.clear()

    def fetch_logs(
        self,
        job_id: str,
        *,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[cluster_pb2.Worker.LogEntry]:
        """Fetch logs for a job, routing to the correct worker.

        Args:
            job_id: Job ID to fetch logs for
            start_ms: Only return logs after this timestamp (exclusive, for incremental polling)
            max_lines: Maximum number of lines to return (0 = unlimited)

        Returns:
            List of LogEntry protos

        Raises:
            ValueError: If job has no worker assigned (not yet scheduled)
        """
        status = self.get_job_status(job_id)
        if not status.worker_address:
            raise ValueError(f"Job {job_id} has no worker assigned (state: {cluster_pb2.JobState.Name(status.state)})")

        worker_client = self._get_worker_client(status.worker_address)
        filter_proto = cluster_pb2.Worker.FetchLogsFilter(
            start_ms=start_ms,
            max_lines=max_lines,
        )
        request = cluster_pb2.Worker.FetchLogsRequest(
            job_id=job_id,
            filter=filter_proto,
        )
        response = worker_client.fetch_logs(request)
        return list(response.logs)
