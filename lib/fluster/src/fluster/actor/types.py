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

"""Core types for the fluster actor layer.

This module contains actor-specific types that depend on the cluster layer.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType

if TYPE_CHECKING:
    from fluster.actor.resolver import Resolver
    from fluster.cluster.types import JobId, Namespace


# Type aliases
ActorId = NewType("ActorId", str)

# Context variable for actor context injection
_actor_context: ContextVar["ActorContext | None"] = ContextVar("actor_context", default=None)


def current_ctx() -> "ActorContext":
    """Get the current ActorContext. Raises if not in an actor call."""
    ctx = _actor_context.get()
    if ctx is None:
        raise RuntimeError("current_ctx() called outside of actor method")
    return ctx


def _set_actor_context(ctx: "ActorContext | None") -> None:
    """Internal: set the actor context for the current call."""
    _actor_context.set(ctx)


@dataclass
class ActorEndpoint:
    """Actor endpoint for discovery and RPC.

    Wraps a cluster Endpoint with actor-specific semantics.

    Args:
        actor_id: Unique actor identifier
        name: Actor name for discovery
        address: Network address (host:port)
        job_id: Job hosting this actor
        namespace: Namespace for scoping
        metadata: Optional key-value metadata
    """

    actor_id: ActorId
    name: str
    address: str
    job_id: "JobId"
    namespace: "Namespace"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ActorContext:
    """Context passed to actor methods as first argument.

    Enables actors to call other actors and access cluster services.

    Args:
        cluster: Cluster client for job management (or None for Stage 1)
        resolver: Resolver for actor discovery (or None for Stage 1)
        job_id: Current job ID
        namespace: Current namespace
    """

    cluster: Any
    resolver: "Resolver | None"
    job_id: str
    namespace: str

    # TODO: Stage 2+: from_environment() will be implemented when ClusterResolver exists
