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

"""Fluster client package.

This package provides the core client interfaces for Fluster:
- ClusterClient, ClusterController: Protocols for cluster operations
- EndpointRegistry: Protocol for actor endpoint registration
- FlusterContext: Execution context available in jobs
- LocalClient: Thread-based local execution
- RpcClusterClient: RPC-based cluster execution
"""

# Import context first - it has no dependencies on actor/resolver
from fluster.client.context import (
    FlusterContext,
    create_context_from_env,
    fluster_ctx,
    fluster_ctx_scope,
    get_fluster_ctx,
)
from fluster.client.protocols import (
    ClusterClient,
    ClusterController,
    EndpointRegistry,
    ResolvedEndpoint,
    ResolveResult,
    Resolver,
)


def __getattr__(name: str):
    """Lazy import for classes that would cause circular imports."""
    if name == "LocalClient":
        from fluster.client.local_client import LocalClient

        return LocalClient
    if name == "LocalClientConfig":
        from fluster.client.local_client import LocalClientConfig

        return LocalClientConfig
    if name == "RpcClusterClient":
        from fluster.client.rpc_client import RpcClusterClient

        return RpcClusterClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClusterClient",
    "ClusterController",
    "EndpointRegistry",
    "FlusterContext",
    "LocalClient",
    "LocalClientConfig",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
    "RpcClusterClient",
    "create_context_from_env",
    "fluster_ctx",
    "fluster_ctx_scope",
    "get_fluster_ctx",
]
