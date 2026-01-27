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

"""Actor server for Fray v2.

Provides a unified ActorServer interface that delegates to backend implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fray.v2.cluster import Cluster


class ActorServer:
    """Actor server for hosting RPC services.

    This is the main interface for hosting actors. It delegates to
    backend-specific implementations (local, Iris, or Ray).

    Example:
        from fray.v2 import current_cluster, ActorServer

        class Counter:
            def __init__(self):
                self.value = 0

            def incr(self, n=1):
                self.value += n
                return self.value

        cluster = current_cluster()
        server = ActorServer(cluster)
        server.register("counter", Counter())
        server.serve()  # Blocks

        # Or for background serving:
        port = server.serve_background()
    """

    def __init__(
        self,
        cluster: Cluster | None = None,
        host: str = "0.0.0.0",
        port: int = 0,
    ):
        """Initialize actor server.

        Args:
            cluster: Cluster for registration. If None, uses current_cluster().
            host: Host address to bind
            port: Port to bind (0 for auto-assignment)
        """
        if cluster is None:
            from fray.v2.cluster import current_cluster

            cluster = current_cluster()

        self._cluster = cluster
        self._host = host
        self._port = port
        self._impl = self._create_impl()

    def _create_impl(self) -> Any:
        """Create backend-specific implementation."""
        # Check if cluster has create_actor_server method
        if hasattr(self._cluster, "create_actor_server"):
            return self._cluster.create_actor_server(self._host, self._port)

        # Fall back to local implementation
        from fray.v2.backends.local import LocalActorServer, LocalCluster

        if isinstance(self._cluster, LocalCluster):
            return LocalActorServer(self._cluster, self._host, self._port)

        raise TypeError(f"Cluster type {type(self._cluster)} does not support ActorServer")

    def register(self, name: str, actor: Any) -> None:
        """Register an actor instance.

        Args:
            name: Name for discovery (will be prefixed with namespace)
            actor: Actor instance (any object with callable methods)
        """
        self._impl.register(name, actor)

    def serve(self) -> None:
        """Start serving requests (blocks indefinitely)."""
        self._impl.serve()

    def serve_background(self) -> int:
        """Start serving in a background thread.

        Returns:
            Port number the server is listening on
        """
        return self._impl.serve_background()

    def shutdown(self) -> None:
        """Shutdown the server."""
        self._impl.shutdown()
