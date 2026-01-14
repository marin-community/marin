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

"""Resolver types and implementations for actor discovery."""

import os
from dataclasses import dataclass, field
from typing import Protocol as TypingProtocol

from fluster import cluster_pb2
from fluster.cluster.types import Namespace
from fluster.cluster_connect import ControllerServiceClientSync


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
    namespace: Namespace
    endpoints: list[ResolvedEndpoint] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.endpoints) == 0

    def first(self) -> ResolvedEndpoint:
        if not self.endpoints:
            raise ValueError(f"No endpoints for '{self.name}' in namespace '{self.namespace}'")
        return self.endpoints[0]


class Resolver(TypingProtocol):
    """Protocol for actor name resolution."""

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult: ...

    @property
    def default_namespace(self) -> Namespace: ...


class FixedResolver:
    """Resolver with statically configured endpoints."""

    def __init__(
        self,
        endpoints: dict[str, str | list[str]],
        namespace: Namespace = Namespace("<local>"),
    ):
        self._namespace = namespace
        self._endpoints: dict[str, list[str]] = {}
        for name, urls in endpoints.items():
            if isinstance(urls, str):
                self._endpoints[name] = [urls]
            else:
                self._endpoints[name] = list(urls)

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        ns = namespace or self._namespace
        urls = self._endpoints.get(name, [])
        endpoints = [ResolvedEndpoint(url=url, actor_id=f"fixed-{name}-{i}") for i, url in enumerate(urls)]
        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)


class ClusterResolver:
    """Resolver backed by the cluster controller's endpoint registry.

    Queries the controller's ListEndpoints RPC to discover actor endpoints
    registered by running jobs. Respects namespace boundaries for isolation.

    Args:
        controller_address: Controller URL (e.g., "http://localhost:8080")
        namespace: Namespace for actor isolation (defaults to FLUSTER_NAMESPACE env var)
        timeout: HTTP request timeout in seconds
    """

    def __init__(
        self,
        controller_address: str,
        namespace: Namespace | None = None,
        timeout: float = 5.0,
    ):
        self._address = controller_address.rstrip("/")
        self._timeout = timeout
        self._namespace = namespace or Namespace(os.environ.get("FLUSTER_NAMESPACE", "<local>"))
        self._client = ControllerServiceClientSync(
            address=self._address,
            timeout_ms=int(timeout * 1000),
        )

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        """Resolve actor name to endpoints via controller.

        Args:
            name: Actor name to resolve
            namespace: Override default namespace

        Returns:
            ResolveResult with matching endpoints
        """
        ns = namespace or self._namespace

        request = cluster_pb2.ListEndpointsRequest(
            prefix=name,
            namespace=str(ns),
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
            if ep.name == name
        ]

        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)


class GcsApi(TypingProtocol):
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

    Discovers actor endpoints by querying GCP VM instance metadata. Instances must
    have metadata tags in the format:
    - `fluster_actor_<name>`: port number for the actor
    - `fluster_namespace`: namespace for isolation (defaults to "<local>")

    Only RUNNING instances are considered for resolution.

    Args:
        project: GCP project ID
        zone: GCP zone (e.g., "us-central1-a")
        namespace: Namespace for actor isolation (defaults to FLUSTER_NAMESPACE env var)
        api: GcsApi implementation (defaults to RealGcsApi)
    """

    ACTOR_PREFIX = "fluster_actor_"
    NAMESPACE_KEY = "fluster_namespace"

    def __init__(
        self,
        project: str,
        zone: str,
        namespace: Namespace | None = None,
        api: GcsApi | None = None,
    ):
        self._project = project
        self._zone = zone
        self._api = api or RealGcsApi()
        self._namespace = namespace or Namespace(os.environ.get("FLUSTER_NAMESPACE", "<local>"))

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        """Resolve actor name to endpoints via GCS instance metadata.

        Args:
            name: Actor name to resolve
            namespace: Override default namespace

        Returns:
            ResolveResult with matching endpoints from RUNNING instances
        """
        ns = namespace or self._namespace
        endpoints = []

        instances = self._api.list_instances(self._project, self._zone)

        for instance in instances:
            if instance.get("status") != "RUNNING":
                continue

            metadata = instance.get("metadata", {})
            instance_ns = metadata.get(self.NAMESPACE_KEY, "<local>")

            if instance_ns != str(ns):
                continue

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

        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)
