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

from fluster.actor.resolver import ResolvedEndpoint, ResolveResult
from fluster.cluster.types import Namespace
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceClientSync


class ClusterResolver:
    """Resolver backed by the cluster controller's endpoint registry.

    Automatically prefixes names with namespace derived from FlusterContext.

    Args:
        controller_address: Controller URL (e.g., "http://localhost:8080")
        timeout: HTTP request timeout in seconds
    """

    def __init__(
        self,
        controller_address: str,
        timeout: float = 5.0,
    ):
        self._address = controller_address.rstrip("/")
        self._timeout = timeout
        self._client = ControllerServiceClientSync(
            address=self._address,
            timeout_ms=int(timeout * 1000),
        )

    def _namespace_prefix(self) -> str:
        from fluster.client.client import get_fluster_ctx

        ctx = get_fluster_ctx()
        if ctx is None:
            raise RuntimeError("No FlusterContext - must be called from within a job")
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
        endpoints = [
            ResolvedEndpoint(
                url=f"http://{ep.address}",
                actor_id=ep.endpoint_id,
                metadata=dict(ep.metadata),
            )
            for ep in resp.endpoints
            if ep.name == prefixed_name
        ]

        return ResolveResult(name=name, endpoints=endpoints)
