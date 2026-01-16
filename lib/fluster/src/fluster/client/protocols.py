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
- Resolver: Protocol for actor name resolution (re-exported from actor.types)

Temporary: ClusterResolver and related implementations are here until Step 3 refactor.
"""

from typing import Protocol, runtime_checkable

from fluster.actor.types import ResolvedEndpoint, ResolveResult, Resolver
from fluster.cluster.types import Entrypoint, JobId, Namespace
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceClientSync

# Re-export from actor.types for backwards compatibility
# TODO(Step 3): Remove these re-exports once all clients import from actor.types
__all__ = [
    "ClusterClient",
    "ClusterResolver",
    "EndpointRegistry",
    "FixedResolver",
    "GcsApi",
    "GcsResolver",
    "MockGcsApi",
    "RealGcsApi",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
]


class FixedResolver:
    """Resolver with statically configured endpoints.

    Used for testing or when endpoints are known ahead of time.
    Does not use namespace prefixing since endpoints are static.
    """

    def __init__(self, endpoints: dict[str, str | list[str]]):
        """Initialize with a mapping of actor names to URLs.

        Args:
            endpoints: Mapping of actor name to URL or list of URLs
        """
        self._endpoints: dict[str, list[str]] = {}
        for name, urls in endpoints.items():
            if isinstance(urls, str):
                self._endpoints[name] = [urls]
            else:
                self._endpoints[name] = list(urls)

    def resolve(self, name: str) -> ResolveResult:
        """Resolve actor name to endpoints.

        Args:
            name: Actor name to resolve

        Returns:
            ResolveResult with configured endpoints
        """
        urls = self._endpoints.get(name, [])
        endpoints = [ResolvedEndpoint(url=url, actor_id=f"fixed-{name}-{i}") for i, url in enumerate(urls)]
        return ResolveResult(name=name, endpoints=endpoints)


class ClusterResolver:
    """Resolver backed by the cluster controller's endpoint registry.

    Queries the controller's ListEndpoints RPC to discover actor endpoints
    registered by running jobs. Automatically prefixes names with the namespace
    derived from the current FlusterContext.

    The namespace prefix is computed dynamically from the current FlusterContext,
    so a single resolver instance can be shared across jobs.

    Args:
        controller_address: Controller URL (e.g., "http://localhost:8080")
        timeout: HTTP request timeout in seconds
    """

    def __init__(
        self,
        controller_address: str,
        timeout: float = 5.0,
    ):
        self._address = controller_address.rstrip("/")
        self._timeout = timeout
        self._client = ControllerServiceClientSync(
            address=self._address,
            timeout_ms=int(timeout * 1000),
        )

    def _namespace_prefix(self) -> str:
        """Get namespace prefix from current FlusterContext."""
        from fluster.client.context import get_fluster_ctx

        ctx = get_fluster_ctx()
        if ctx is None:
            raise RuntimeError("No FlusterContext - must be called from within a job")
        return str(Namespace.from_job_id(ctx.job_id))

    def resolve(self, name: str) -> ResolveResult:
        """Resolve actor name to endpoints via controller.

        The name is auto-prefixed with the namespace before lookup.
        For example, resolving "calculator" with namespace "abc123" looks up
        "abc123/calculator".

        Args:
            name: Actor name to resolve (will be prefixed with namespace)

        Returns:
            ResolveResult with matching endpoints
        """
        prefixed_name = f"{self._namespace_prefix()}/{name}"

        request = cluster_pb2.Controller.ListEndpointsRequest(
            prefix=prefixed_name,
        )

        resp = self._client.list_endpoints(request)

        # Filter to exact name matches (controller uses prefix matching)
        endpoints = [
            ResolvedEndpoint(
                url=f"http://{ep.address}",
                actor_id=ep.endpoint_id,
                metadata=dict(ep.metadata),
            )
            for ep in resp.endpoints
            if ep.name == prefixed_name
        ]

        return ResolveResult(name=name, endpoints=endpoints)


class GcsApi(Protocol):
    """Protocol for GCS Compute API operations."""

    def list_instances(self, project: str, zone: str) -> list[dict]:
        """List VM instances with metadata."""
        ...


class RealGcsApi:
    """Real GCS API using google-cloud-compute."""

    def list_instances(self, project: str, zone: str) -> list[dict]:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        instances = []
        for instance in client.list(project=project, zone=zone):
            metadata = {}
            if instance.metadata and instance.metadata.items:
                for item in instance.metadata.items:
                    metadata[item.key] = item.value

            internal_ip = None
            if instance.network_interfaces:
                internal_ip = instance.network_interfaces[0].network_i_p

            instances.append(
                {
                    "name": instance.name,
                    "internal_ip": internal_ip,
                    "metadata": metadata,
                    "status": instance.status,
                }
            )
        return instances


class MockGcsApi:
    """Mock GCS API for testing."""

    def __init__(self, instances: list[dict] | None = None):
        self._instances = instances or []

    def set_instances(self, instances: list[dict]) -> None:
        self._instances = instances

    def list_instances(self, project: str, zone: str) -> list[dict]:
        return self._instances


class GcsResolver:
    """Resolver using GCS VM instance metadata tags.

    Discovers actor endpoints by querying GCP VM instance metadata. This is purely
    an infrastructure discovery mechanism - it finds all running instances with
    matching actor metadata.

    Instances must have metadata tags in the format:
    - `fluster_actor_<name>`: port number for the actor

    Only RUNNING instances are considered for resolution.

    Note: Unlike ClusterResolver, GcsResolver does NOT do namespace prefixing.
    It's an infrastructure discovery mechanism that returns all instances with
    matching actor metadata tags. Use this for static VM-based deployments where
    namespace isolation is not needed.

    Args:
        project: GCP project ID
        zone: GCP zone (e.g., "us-central1-a")
        api: GcsApi implementation (defaults to RealGcsApi)
    """

    ACTOR_PREFIX = "fluster_actor_"

    def __init__(
        self,
        project: str,
        zone: str,
        api: GcsApi | None = None,
    ):
        self._project = project
        self._zone = zone
        self._api = api or RealGcsApi()

    def resolve(self, name: str) -> ResolveResult:
        """Resolve actor name to endpoints via GCS instance metadata.

        Discovers all running instances with matching actor metadata.
        No namespace prefixing is applied.

        Args:
            name: Actor name to resolve

        Returns:
            ResolveResult with matching endpoints from RUNNING instances
        """
        endpoints = []

        instances = self._api.list_instances(self._project, self._zone)

        for instance in instances:
            if instance.get("status") != "RUNNING":
                continue

            metadata = instance.get("metadata", {})

            actor_key = f"{self.ACTOR_PREFIX}{name}"
            if actor_key in metadata:
                port = metadata[actor_key]
                ip = instance.get("internal_ip")
                if ip:
                    endpoints.append(
                        ResolvedEndpoint(
                            url=f"http://{ip}:{port}",
                            actor_id=f"gcs-{instance['name']}-{name}",
                            metadata={"instance": instance["name"]},
                        )
                    )

        return ResolveResult(name=name, endpoints=endpoints)


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
