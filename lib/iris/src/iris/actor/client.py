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

"""Actor client for making RPC calls to actor servers.

The ActorClient provides transparent actor discovery and invocation with
automatic retry logic. When an actor name cannot be resolved immediately
(e.g., actor server still starting), the client retries with exponential
backoff until the timeout is reached.

Typical actor startup: 2-8 seconds (Docker container + server initialization).
Default timeout (30s) handles this gracefully.

Example:
    resolver = ClusterResolver("http://controller:8080")
    client = ActorClient(resolver, "my-actor")
    result = client.some_method(arg1, arg2)  # Retries until actor found

Custom backoff behavior:
    client = ActorClient(
        resolver, "my-actor",
        initial_backoff=0.2, max_backoff=5.0,
    )
"""

import logging
import time
from typing import Any

import cloudpickle

from iris.actor.resolver import Resolver, ResolveResult
from iris.rpc import actor_pb2
from iris.rpc.actor_connect import ActorServiceClientSync
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)


class ActorClient:
    """Actor client with resolver-based discovery."""

    def __init__(
        self,
        resolver: Resolver,
        name: str,
        timeout: float = 30.0,
        call_timeout: float | None = None,
        initial_backoff: float = 0.1,
        max_backoff: float = 2.0,
        backoff_factor: float = 2.0,
        backoff_jitter: float = 0.25,
    ):
        """Initialize the actor client.

        Args:
            resolver: Resolver instance for endpoint discovery
            name: Name of the actor to invoke
            timeout: Total timeout in seconds for resolution + RPC calls.
                When resolving, retries continue until this timeout is reached.
            call_timeout: Timeout in seconds for RPC calls. Defaults to `timeout`
                when not specified.
            initial_backoff: Initial retry delay in seconds
            max_backoff: Maximum delay between retries in seconds
            backoff_factor: Multiplier for exponential backoff
            backoff_jitter: Random jitter as fraction of delay (e.g., 0.25 = ±25%)
        """
        self._resolver = resolver
        self._name = name
        self._timeout = timeout
        self._call_timeout = timeout if call_timeout is None else call_timeout
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_factor = backoff_factor
        self._backoff_jitter = backoff_jitter
        self._cached_result: ResolveResult | None = None
        self._client: ActorServiceClientSync | None = None
        self._client_url: str | None = None

    def _resolve(self) -> ResolveResult:
        """Resolve actor name with exponential backoff retry.

        Returns:
            ResolveResult with at least one endpoint

        Raises:
            TimeoutError: If no endpoints found within timeout
        """
        if self._cached_result is not None and not self._cached_result.is_empty:
            return self._cached_result

        backoff = ExponentialBackoff(
            initial=self._initial_backoff,
            maximum=self._max_backoff,
            factor=self._backoff_factor,
            jitter=self._backoff_jitter,
        )
        start_time = time.monotonic()
        attempt = 0

        while True:
            result = self._resolver.resolve(self._name)

            if not result.is_empty:
                self._cached_result = result
                if attempt > 0:
                    logger.debug(
                        f"Resolved actor '{self._name}' to {len(result.endpoints)} endpoint(s) "
                        f"after {attempt} retries in {time.monotonic() - start_time:.2f}s"
                    )
                return result

            elapsed = time.monotonic() - start_time
            if elapsed >= self._timeout:
                raise TimeoutError(f"Failed to resolve actor '{self._name}' after {self._timeout}s ({attempt} retries)")

            delay = backoff.next_interval()
            remaining = self._timeout - elapsed
            delay = min(delay, remaining)

            if delay > 0:
                logger.debug(
                    f"Actor '{self._name}' not found, retrying in {delay:.3f}s "
                    f"(attempt {attempt + 1}, elapsed {elapsed:.2f}s/{self._timeout}s)"
                )
                time.sleep(delay)

            attempt += 1

    def _invalidate_cache(self) -> None:
        self._cached_result = None
        self._client = None
        self._client_url = None

    def _get_client(self, url: str) -> ActorServiceClientSync:
        if self._client is None or self._client_url != url:
            self._client = ActorServiceClientSync(
                address=url,
                timeout_ms=None if self._call_timeout is None else int(self._call_timeout * 1000),
            )
            self._client_url = url
        return self._client

    def __getattr__(self, method_name: str) -> "_RpcMethod":
        return _RpcMethod(self, method_name)


class _RpcMethod:
    def __init__(self, client: ActorClient, method_name: str):
        self._client = client
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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
