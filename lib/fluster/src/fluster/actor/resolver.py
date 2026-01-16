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

"""Core types and resolver implementations for the actor system.

This module defines the fundamental types used throughout the actor system:
- Resolver: Protocol for actor name resolution
- ResolvedEndpoint: A single resolved endpoint for an actor
- ResolveResult: Result containing one or more endpoints

Resolver implementations:
- FixedResolver: Static endpoint configuration (for testing)
- GcsResolver: Discovery via GCP VM instance metadata tags
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ResolvedEndpoint:
    """A single resolved endpoint for an actor.

    Attributes:
        url: Full URL to the actor endpoint (e.g., "http://host:port")
        actor_id: Unique identifier for this actor instance
        metadata: Optional metadata associated with the endpoint
    """

    url: str
    actor_id: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolveResult:
    """Result of resolving an actor name to endpoints.

    Attributes:
        name: The actor name that was resolved
        endpoints: List of resolved endpoints (empty if not found)
    """

    name: str
    endpoints: list[ResolvedEndpoint] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Returns True if no endpoints were found."""
        return len(self.endpoints) == 0

    def first(self) -> ResolvedEndpoint:
        """Get the first endpoint.

        Returns:
            The first resolved endpoint

        Raises:
            ValueError: If no endpoints are available
        """
        if not self.endpoints:
            raise ValueError(f"No endpoints for '{self.name}'")
        return self.endpoints[0]


class Resolver(Protocol):
    """Protocol for resolving actor names to endpoints.

    Implementations of this protocol discover actor endpoints by name.
    The resolver is responsible for any namespace prefixing or discovery logic.

    Implementations:
    - FixedResolver: Static endpoint mapping
    - GcsResolver: Discovers via GCS VM metadata
    - ClusterResolver: Resolves via cluster controller (lives in fluster.client)
    """

    def resolve(self, name: str) -> ResolveResult:
        """Resolve an actor name to endpoints.

        Args:
            name: Actor name to resolve

        Returns:
            ResolveResult with zero or more endpoints
        """
        ...


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
