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

"""Core types and protocols for the actor system.

This module defines the fundamental types used throughout the actor system:
- Resolver: Protocol for actor name resolution
- ResolvedEndpoint: A single resolved endpoint for an actor
- ResolveResult: Result containing one or more endpoints

These are low-level actor concepts - implementations with context magic live in fluster.client.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class Resolver(Protocol):
    """Protocol for resolving actor names to endpoints.

    Implementations of this protocol discover actor endpoints by name.
    The resolver is responsible for any namespace prefixing or discovery logic.

    Implementations:
    - ClusterResolver: Resolves via cluster controller (lives in fluster.client)
    - LocalResolver: Resolves via local endpoint store (lives in fluster.client)
    - FixedResolver: Static endpoint mapping (lives in fluster.client)
    - GcsResolver: Discovers via GCS VM metadata (lives in fluster.client)
    """

    def resolve(self, name: str) -> ResolveResult:
        """Resolve an actor name to endpoints.

        Args:
            name: Actor name to resolve

        Returns:
            ResolveResult with zero or more endpoints
        """
        ...
