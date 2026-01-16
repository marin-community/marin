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

"""Protocols for cluster operations.

This module defines the ClusterOperations protocol, which provides raw cluster
operations with explicit parameters. Implementations include:
- RpcClusterOperations: talks to controller via RPC
- LocalClusterOperations: in-process thread-based execution

These are "dumb" operations - all parameters are explicit, no context magic.
"""

from typing import Protocol, runtime_checkable

from fluster.cluster.types import Entrypoint
from fluster.rpc import cluster_pb2


@runtime_checkable
class ClusterOperations(Protocol):
    """Raw cluster operations - all params explicit, no namespace magic.

    This protocol defines the low-level cluster operations interface.
    Implementations should:
    - Take fully-qualified job IDs (not relative names)
    - Take fully-qualified endpoint names (not auto-prefixed)
    - Not use context lookups for implicit parameters

    For high-level operations with context magic, use fluster.client.ClusterClient.
    """

    def submit_job(
        self,
        job_id: str,
        entrypoint: Entrypoint,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> None:
        """Submit a job with explicit full job ID.

        Args:
            job_id: Full hierarchical job ID (e.g., "root/worker-0")
            entrypoint: Job entrypoint (callable + args/kwargs)
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])
        """
        ...

    def get_job_status(self, job_id: str) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Full job ID

        Returns:
            JobStatus proto with current state
        """
        ...

    def wait_for_job(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Full job ID
            timeout: Maximum time to wait in seconds
            poll_interval: Maximum time between status checks

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        ...

    def terminate_job(self, job_id: str) -> None:
        """Terminate a running job.

        Args:
            job_id: Full job ID
        """
        ...

    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint.

        Args:
            name: Full endpoint name (with namespace prefix if needed)
            address: Address where actor is listening (host:port)
            job_id: Job ID that owns this endpoint
            metadata: Optional metadata

        Returns:
            Endpoint ID assigned by the system
        """
        ...

    def unregister_endpoint(self, endpoint_id: str) -> None:
        """Unregister an endpoint.

        Args:
            endpoint_id: Endpoint ID to remove
        """
        ...

    def list_endpoints(self, prefix: str) -> list[cluster_pb2.Controller.Endpoint]:
        """List endpoints matching a prefix.

        Args:
            prefix: Name prefix to match (e.g., "abc123/")

        Returns:
            List of matching endpoints
        """
        ...
