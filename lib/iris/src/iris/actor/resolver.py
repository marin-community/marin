# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core types and resolver implementations for the actor system."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ResolvedEndpoint:
    """A single resolved endpoint for an actor."""

    url: str
    actor_id: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolveResult:
    """Result of resolving an actor name to endpoints."""

    name: str
    endpoints: list[ResolvedEndpoint] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
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

    Implementations:
    - FixedResolver: Static endpoint mapping
    - ClusterResolver: Resolves via cluster controller (lives in iris.client)
    """

    def resolve(self, name: str) -> ResolveResult: ...


class FixedResolver:
    """Resolver with statically configured endpoints.

    Used for testing or when endpoints are known ahead of time.
    Does not use namespace prefixing since endpoints are static.
    """

    def __init__(self, endpoints: dict[str, str | list[str]]):
        """Initialize with a mapping of actor names to URLs."""
        self._endpoints: dict[str, list[str]] = {}
        for name, urls in endpoints.items():
            if isinstance(urls, str):
                self._endpoints[name] = [urls]
            else:
                self._endpoints[name] = list(urls)

    def resolve(self, name: str) -> ResolveResult:
        urls = self._endpoints.get(name, [])
        endpoints = [ResolvedEndpoint(url=url, actor_id=f"fixed-{name}-{i}") for i, url in enumerate(urls)]
        return ResolveResult(name=name, endpoints=endpoints)


class ProxyResolver:
    """Resolver that routes actor calls through the controller's endpoint proxy.

    Instead of resolving to the actor's direct address, builds a per-actor
    base URL of the form ``<controller>/proxy/<encoded-name>`` where the
    actor name has its leading slash stripped and remaining slashes replaced
    with dots. The Connect transport appends the service/method path, so a
    call to ``iris.actor.ActorService/Call`` reaches:

        ``<controller>/proxy/<encoded-name>/iris.actor.ActorService/Call``

    The native listener receives ``encoded-name`` as the path parameter and
    resolves it back to the actor's registered address with dot → slash
    substitution.

    Args:
        controller_url: Controller URL (e.g., ``http://localhost:8080``)
    """

    def __init__(self, controller_url: str):
        self._controller_url = controller_url.rstrip("/")

    def resolve(self, name: str) -> ResolveResult:
        encoded = name.lstrip("/").replace("/", ".")
        return ResolveResult(
            name=name,
            endpoints=[
                ResolvedEndpoint(
                    url=f"{self._controller_url}/proxy/{encoded}",
                    actor_id=f"proxy-{name}",
                )
            ],
        )
