# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor client for making RPC calls to actor servers.

The ActorClient provides transparent actor discovery and invocation with
automatic retry logic. When an actor name cannot be resolved immediately
(e.g., actor server still starting), the client retries with exponential
backoff until the timeout is reached.

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

from iris.actor.resolver import Resolver
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
        resolve_timeout: float = 3600.0,
        call_timeout: float | None = None,
        initial_backoff: float = 0.1,
        max_backoff: float = 10.0,
        backoff_factor: float = 2.0,
        backoff_jitter: float = 0.25,
    ):
        """Initialize the actor client.

        Args:
            resolver: Resolver instance for endpoint discovery
            name: Name of the actor to invoke
            resolve_timeout: Total timeout in seconds for initial worker resolution.
            call_timeout: Timeout in seconds for RPC calls. Defaults to `timeout`
                when not specified.
            initial_backoff: Initial retry delay in seconds
            max_backoff: Maximum delay between retries in seconds
            backoff_factor: Multiplier for exponential backoff
            backoff_jitter: Random jitter as fraction of delay (e.g., 0.25 = ±25%)
        """
        self._resolver = resolver
        self._name = name
        self._resolve_timeout = resolve_timeout
        self._call_timeout = resolve_timeout if call_timeout is None else call_timeout
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_factor = backoff_factor
        self._backoff_jitter = backoff_jitter
        self._rpc_client: ActorServiceClientSync | None = None

    def rpc_client(self) -> ActorServiceClientSync:
        """Resolve actor name with exponential backoff retry.

        Returns:
            ResolveResult with at least one endpoint

        Raises:
            TimeoutError: If no endpoints found within timeout
        """
        if self._rpc_client:
            return self._rpc_client

        backoff = ExponentialBackoff(
            initial=self._initial_backoff,
            maximum=self._max_backoff,
            factor=self._backoff_factor,
            jitter=self._backoff_jitter,
        )
        start_time = time.monotonic()
        attempt = 0

        while True:
            logger.info("Resolving name %s via %s", self._name, self._resolver)
            result = self._resolver.resolve(self._name)

            if not result.is_empty:
                logger.info(
                    f"Resolved actor '{self._name}' to {len(result.endpoints)} endpoint(s) "
                    f"after {attempt} retries in {time.monotonic() - start_time:.2f}s"
                )
                url = result.first().url
                self._rpc_client = ActorServiceClientSync(
                    address=url,
                    timeout_ms=None if self._call_timeout is None else int(self._call_timeout * 1000),
                )
                return self._rpc_client

            elapsed = time.monotonic() - start_time
            if elapsed >= self._resolve_timeout:
                raise TimeoutError(
                    f"Failed to resolve actor '{self._name}' after {self._resolve_timeout}s ({attempt} retries)"
                )

            delay = backoff.next_interval()
            remaining = self._resolve_timeout - elapsed
            delay = min(delay, remaining)

            if delay > 0:
                logger.debug(
                    f"Actor '{self._name}' not found, retrying in {delay:.3f}s "
                    f"(attempt {attempt + 1}, elapsed {elapsed:.2f}s/{self._resolve_timeout}s)"
                )
                time.sleep(delay)

            attempt += 1

    def __getattr__(self, method_name: str) -> "_RpcMethod":
        return _RpcMethod(self, method_name)


class _RpcMethod:
    def __init__(self, client: ActorClient, method_name: str):
        self._client = client
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call = actor_pb2.ActorCall(
            method_name=self._method_name,
            actor_name=self._client._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        try:
            client = self._client.rpc_client()
            resp = client.call(call)
        except Exception:
            self._client._rpc_client = None
            raise

        if resp.HasField("error"):
            if resp.error.serialized_exception:
                raise cloudpickle.loads(resp.error.serialized_exception)
            raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

        return cloudpickle.loads(resp.serialized_value)
