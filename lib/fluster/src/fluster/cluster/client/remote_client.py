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

"""RPC-based cluster client implementation.

This module provides RemoteClusterClient, which implements the ClusterClient
protocol by talking to the controller via RPC.
"""

import time

import cloudpickle

from fluster.cluster.types import Entrypoint, is_job_finished
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceClientSync
from fluster.time_utils import ExponentialBackoff


class RemoteClusterClient:
    """Cluster client via RPC to controller.

    This is a "dumb" implementation - all parameters are explicit, no context magic.
    Takes full job IDs, full endpoint names, etc.
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

    def submit_job(
        self,
        job_id: str,
        entrypoint: Entrypoint,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
    ) -> None:
        """Submit a job to the cluster via RPC.

        Args:
            job_id: Full hierarchical job ID (e.g., "root/worker-0")
            entrypoint: Job entrypoint (callable + args/kwargs)
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])
            scheduling_timeout_seconds: Maximum time to wait for scheduling (0 = no timeout)
        """
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
        """Get job status via RPC.

        Args:
            job_id: Full job ID

        Returns:
            JobStatus proto with current state
        """
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
        """Terminate a running job via RPC.

        Args:
            job_id: Full job ID
        """
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self._client.terminate_job(request)

    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint via RPC.

        Args:
            name: Full endpoint name (with namespace prefix if needed)
            address: Address where actor is listening (host:port)
            job_id: Job ID that owns this endpoint
            metadata: Optional metadata

        Returns:
            Endpoint ID assigned by controller
        """
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
        """List endpoints matching a prefix via RPC.

        Args:
            prefix: Name prefix to match (e.g., "abc123/")

        Returns:
            List of matching endpoints
        """
        request = cluster_pb2.Controller.ListEndpointsRequest(prefix=prefix)
        response = self._client.list_endpoints(request)
        return list(response.endpoints)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the client.

        No-op for RemoteClusterClient - the RPC client doesn't hold resources.

        Args:
            wait: Ignored
        """
        del wait
