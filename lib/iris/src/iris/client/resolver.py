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

"""Namespace-aware resolver for actor discovery via cluster controller."""

import os

from iris.actor.resolver import ResolvedEndpoint, ResolveResult
from iris.cluster.types import Namespace
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync


def _rewrite_address_for_host(address: str) -> str:
    """Rewrite Docker-internal addresses to localhost when running on the host.

    When actors register from inside Docker containers, they use host.docker.internal
    so other containers can reach them. But clients running on the host (outside Docker)
    can't resolve host.docker.internal - they should use localhost instead since Docker
    publishes ports to the host.

    This function checks if we're running outside Docker (no IRIS_ADVERTISE_HOST set)
    and rewrites host.docker.internal to localhost.
    """
    # If IRIS_ADVERTISE_HOST is set, we're inside a container and should use Docker networking
    if os.environ.get("IRIS_ADVERTISE_HOST"):
        return address
    # Otherwise, we're on the host - rewrite Docker-internal addresses
    return address.replace("host.docker.internal", "localhost")


class ClusterResolver:
    """Resolver backed by the cluster controller's endpoint registry.

    Can be used in two modes:
    1. With explicit namespace: Use directly from client code without IrisContext
    2. Without explicit namespace: Derives namespace from IrisContext (requires job context)

    Args:
        controller_address: Controller URL (e.g., "http://localhost:8080")
        timeout: HTTP request timeout in seconds
        namespace: Optional explicit namespace (if not provided, uses IrisContext)
    """

    def __init__(
        self,
        controller_address: str,
        timeout: float = 5.0,
        namespace: Namespace | None = None,
    ):
        self._address = controller_address.rstrip("/")
        self._timeout = timeout
        self._explicit_namespace = namespace
        self._client = ControllerServiceClientSync(
            address=self._address,
            timeout_ms=int(timeout * 1000),
        )

    def _namespace_prefix(self) -> str:
        if self._explicit_namespace is not None:
            return str(self._explicit_namespace)
        from iris.client.client import get_iris_ctx

        ctx = get_iris_ctx()
        if ctx is None:
            raise RuntimeError("No IrisContext - provide explicit namespace or call from within a job")
        if ctx.job_id is None:
            raise RuntimeError("No job id available - ensure IrisContext is initialized from a job")
        return str(Namespace.from_job_id(ctx.job_id))

    def resolve(self, name: str) -> ResolveResult:
        """Resolve actor name to endpoints via controller.

        Args:
            name: Actor name to resolve (will be prefixed with namespace)

        Returns:
            ResolveResult with matching endpoints
        """
        prefixed_name = f"{self._namespace_prefix()}/{name}"

        request = cluster_pb2.Controller.ListEndpointsRequest(
            prefix=prefixed_name,
        )

        resp = self._client.list_endpoints(request)

        # Filter to exact name matches (controller uses prefix matching)
        # Rewrite addresses for host/container compatibility
        endpoints = [
            ResolvedEndpoint(
                url=f"http://{_rewrite_address_for_host(ep.address)}",
                actor_id=ep.endpoint_id,
                metadata=dict(ep.metadata),
            )
            for ep in resp.endpoints
            if ep.name == prefixed_name
        ]

        return ResolveResult(name=name, endpoints=endpoints)
