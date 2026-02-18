# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor system for distributed RPC.

For ClusterResolver (namespace-aware controller-based resolution),
see iris.client.resolver.
"""

from iris.actor.client import ActorClient
from iris.actor.pool import ActorPool, BroadcastFuture, CallResult
from iris.actor.resolver import (
    FixedResolver,
    GcsApi,
    GcsResolver,
    MockGcsApi,
    RealGcsApi,
    ResolvedEndpoint,
    ResolveResult,
    Resolver,
)
from iris.actor.server import ActorId, ActorServer

__all__ = [
    "ActorClient",
    "ActorId",
    "ActorPool",
    "ActorServer",
    "BroadcastFuture",
    "CallResult",
    "FixedResolver",
    "GcsApi",
    "GcsResolver",
    "MockGcsApi",
    "RealGcsApi",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
]
