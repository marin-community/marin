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

"""Actor system for distributed RPC.

This module provides the core actor infrastructure:
- ActorServer: Host actor instances and handle RPC calls
- ActorClient: Call remote actors via a resolver
- ActorPool: Load-balanced and broadcast calls to multiple actors
- Resolver: Protocol for actor name resolution
- FixedResolver: Static endpoint configuration (for testing)
- GcsResolver: Discovery via GCP VM instance metadata

For ClusterResolver (namespace-aware controller-based resolution),
see fluster.client.resolver.
"""

from fluster.actor.client import ActorClient, RetryConfig
from fluster.actor.pool import ActorPool, BroadcastFuture, CallResult
from fluster.actor.resolver import (
    FixedResolver,
    GcsApi,
    GcsResolver,
    MockGcsApi,
    RealGcsApi,
    ResolvedEndpoint,
    ResolveResult,
    Resolver,
)
from fluster.actor.server import ActorId, ActorServer

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
    "RetryConfig",
]
