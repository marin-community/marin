# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor system for distributed RPC.

For ClusterResolver (namespace-aware controller-based resolution),
see iris.client.resolver.
"""

from iris.actor.client import ActorClient
from iris.actor.pool import ActorPool, BroadcastFuture, CallResult
from iris.actor.resolver import (
    ACTOR_ENDPOINT_HEADER,
    FixedResolver,
    ProxyResolver,
    ResolvedEndpoint,
    Resolver,
    ResolveResult,
)
from iris.actor.server import ActorId, ActorServer

__all__ = [
    "ACTOR_ENDPOINT_HEADER",
    "ActorClient",
    "ActorId",
    "ActorPool",
    "ActorServer",
    "BroadcastFuture",
    "CallResult",
    "FixedResolver",
    "ProxyResolver",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
]
