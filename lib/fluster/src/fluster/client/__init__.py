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

"""High-level client layer with automatic namespace management and job hierarchy."""

from fluster.actor.resolver import (
    FixedResolver,
    GcsResolver,
    ResolvedEndpoint,
    ResolveResult,
    Resolver,
)
from fluster.client.client import (
    EndpointRegistry,
    FlusterClient,
    FlusterContext,
    LocalClientConfig,
    LogEntry,
    create_context_from_env,
    fluster_ctx,
    fluster_ctx_scope,
    get_fluster_ctx,
)
from fluster.client.log_poller import LogPoller
from fluster.client.resolver import ClusterResolver
from fluster.client.worker_pool import WorkerPool, WorkerPoolConfig

__all__ = [
    "ClusterResolver",
    "EndpointRegistry",
    "FixedResolver",
    "FlusterClient",
    "FlusterContext",
    "GcsResolver",
    "LocalClientConfig",
    "LogEntry",
    "LogPoller",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
    "WorkerPool",
    "WorkerPoolConfig",
    "create_context_from_env",
    "fluster_ctx",
    "fluster_ctx_scope",
    "get_fluster_ctx",
]
