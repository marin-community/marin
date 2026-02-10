# Copyright 2025 The Marin Authors
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
    - GcsResolver: Discovers via GCS VM metadata
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


class GcsApi(Protocol):
    def list_instances(self, project: str, zone: str) -> list[dict]: ...


class RealGcsApi:
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
    def __init__(self, instances: list[dict] | None = None):
        self._instances = instances or []

    def set_instances(self, instances: list[dict]) -> None:
        self._instances = instances

    def list_instances(self, project: str, zone: str) -> list[dict]:
        return self._instances


class GcsResolver:
    """Resolver using GCS VM instance metadata tags.

    Discovers actor endpoints by querying GCP VM instance metadata. Unlike
    ClusterResolver, this does NOT do namespace prefixing. Use this for
    static VM-based deployments where namespace isolation is not needed.

    Instances must have metadata tags: `iris_actor_<name>` = port number.
    Only RUNNING instances are considered.
    """

    ACTOR_PREFIX = "iris_actor_"

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
