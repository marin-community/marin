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

"""Actor client for making RPC calls to actor servers."""

from typing import Any

import cloudpickle

from fluster import actor_pb2
from fluster.actor.resolver import Resolver, ResolveResult
from fluster.actor_connect import ActorServiceClientSync


class ActorClient:
    """Actor client with resolver-based discovery."""

    def __init__(
        self,
        resolver: Resolver,
        name: str,
        timeout: float = 30.0,
    ):
        """Initialize the actor client.

        Args:
            resolver: Resolver instance for endpoint discovery
            name: Name of the actor to invoke
            timeout: Request timeout in seconds
        """
        self._resolver = resolver
        self._name = name
        self._timeout = timeout
        self._cached_result: ResolveResult | None = None
        self._client: ActorServiceClientSync | None = None
        self._client_url: str | None = None

    def _resolve(self) -> ResolveResult:
        if self._cached_result is None or self._cached_result.is_empty:
            self._cached_result = self._resolver.resolve(self._name)
        return self._cached_result

    def _invalidate_cache(self) -> None:
        self._cached_result = None
        self._client = None
        self._client_url = None

    def _get_client(self, url: str) -> ActorServiceClientSync:
        """Get or create a client for the given URL."""
        if self._client is None or self._client_url != url:
            self._client = ActorServiceClientSync(
                address=url,
                timeout_ms=int(self._timeout * 1000),
            )
            self._client_url = url
        return self._client

    def __getattr__(self, method_name: str) -> "_RpcMethod":
        return _RpcMethod(self, method_name)


class _RpcMethod:
    """Represents a single RPC method call."""

    def __init__(self, client: ActorClient, method_name: str):
        self._client = client
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the RPC call."""
        result = self._client._resolve()
        if result.is_empty:
            raise RuntimeError(f"No endpoints found for actor '{self._client._name}'")

        endpoint = result.first()

        call = actor_pb2.ActorCall(
            method_name=self._method_name,
            actor_name=self._client._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        try:
            client = self._client._get_client(endpoint.url)
            resp = client.call(call)
        except Exception:
            self._client._invalidate_cache()
            raise

        if resp.HasField("error"):
            if resp.error.serialized_exception:
                raise cloudpickle.loads(resp.error.serialized_exception)
            raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

        return cloudpickle.loads(resp.serialized_value)
