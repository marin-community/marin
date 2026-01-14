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

"""Actor system for distributed RPC."""

from fluster.actor.client import ActorClient
from fluster.actor.pool import ActorPool, BroadcastFuture, CallResult
from fluster.actor.resolver import (
    ClusterResolver,
    FixedResolver,
    ResolveResult,
    ResolvedEndpoint,
    Resolver,
)
from fluster.actor.server import ActorServer
from fluster.actor.types import ActorContext, ActorId, current_ctx

__all__ = [
    "ActorClient",
    "ActorContext",
    "ActorId",
    "ActorPool",
    "ActorServer",
    "BroadcastFuture",
    "CallResult",
    "ClusterResolver",
    "FixedResolver",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
    "current_ctx",
]
