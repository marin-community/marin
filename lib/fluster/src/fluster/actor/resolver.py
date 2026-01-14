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
from typing import Protocol

import httpx

from fluster import cluster_pb2
from fluster.cluster.types import Namespace


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


class Resolver(Protocol):
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

        response = httpx.post(
            f"{self._address}/fluster.cluster.ControllerService/ListEndpoints",
            content=request.SerializeToString(),
            headers={"Content-Type": "application/proto"},
            timeout=self._timeout,
        )
        response.raise_for_status()

        resp = cluster_pb2.ListEndpointsResponse()
        resp.ParseFromString(response.content)

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
