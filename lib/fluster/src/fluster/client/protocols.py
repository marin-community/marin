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

"""Protocol definitions for Fluster client interfaces.

This module defines the core protocols used for cluster operations and endpoint
registration. These protocols enable the same job code to work transparently
with both local execution (LocalClient) and remote cluster execution (RpcClusterClient).

Protocols:
- EndpointRegistry: Registering actor endpoints for discovery
- ClusterClient: Cluster client interface for job submission
- Resolver: Protocol for actor name resolution
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from fluster.cluster.types import Entrypoint, JobId
from fluster.rpc import cluster_pb2


@dataclass
class ResolvedEndpoint:
    """A single resolved endpoint."""

    url: str  # e.g., "http://host:port"
    actor_id: str  # Unique handle for staleness detection
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolveResult:
    """Result of resolving an actor name."""

    name: str
    endpoints: list[ResolvedEndpoint] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.endpoints) == 0

    def first(self) -> ResolvedEndpoint:
        if not self.endpoints:
            raise ValueError(f"No endpoints for '{self.name}'")
        return self.endpoints[0]


@runtime_checkable
class Resolver(Protocol):
    """Protocol for actor name resolution.

    Resolvers automatically handle namespace prefixing based on the current
    FlusterContext. When resolving "calculator", the resolver will look up
    "{namespace}/calculator" where namespace is derived from the context.

    Implementations:
    - ClusterResolver: Backed by the cluster controller's endpoint registry
    - LocalResolver: Backed by LocalClient's in-memory endpoint store
    - FixedResolver: Statically configured endpoints
    - GcsResolver: Discovery via GCS VM instance metadata
    """

    def resolve(self, name: str) -> ResolveResult: ...


@runtime_checkable
class EndpointRegistry(Protocol):
    """Protocol for registering actor endpoints.

    Implementations handle namespace prefixing automatically based on job context.

    Implementations:
    - RpcEndpointRegistry: Registers via RPC to controller
    - LocalEndpointRegistry: Per-job wrapper around _EndpointStore
    """

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint for actor discovery.

        Args:
            name: Actor name for discovery
            address: Address where actor is listening (host:port)
            metadata: Optional metadata for the endpoint

        Returns:
            Unique endpoint ID for later unregistration
        """
        ...

    def unregister(self, endpoint_id: str) -> None:
        """Unregister a previously registered endpoint.

        Args:
            endpoint_id: ID returned from register()
        """
        ...


@runtime_checkable
class ClusterClient(Protocol):
    """Protocol for cluster job operations.

    This is the interface WorkerPool and other clients use to interact
    with a cluster.

    Implementations:
    - RpcClusterClient: Backed by RPC to controller
    - LocalClient: In-process thread-based execution
    """

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job to the cluster.

        The name is combined with the current job's name (if any) to form a
        hierarchical identifier. For example, if the current job is "my-exp"
        and you submit with name="worker-0", the full name becomes "my-exp/worker-0".

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """
        ...

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobStatus proto with current state
        """
        ...

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        ...

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Args:
            job_id: Job ID to terminate
        """
        ...

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        The namespace is derived from the current job context.

        Returns:
            Resolver instance
        """
        ...
