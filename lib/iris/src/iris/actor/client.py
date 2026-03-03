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
        backoff=ExponentialBackoff(initial=0.2, maximum=5.0),
        max_call_attempts=3,
    )
"""

import logging
from typing import Any

import cloudpickle
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.actor.resolver import Resolver
from iris.rpc import actor_pb2
from iris.rpc.actor_connect import ActorServiceClientSync
from iris.rpc.errors import call_with_retry
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)


def unwrap_actor_response(resp: actor_pb2.ActorResponse) -> Any:
    """Unwrap an ActorResponse, raising the embedded exception on error."""
    if resp.HasField("error"):
        if resp.error.serialized_exception:
            raise cloudpickle.loads(resp.error.serialized_exception)
        raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")
    return cloudpickle.loads(resp.serialized_value)


class ActorClient:
    """Actor client with resolver-based discovery."""

    def __init__(
        self,
        resolver: Resolver,
        name: str,
        resolve_timeout: float = 3600.0,
        call_timeout: float | None = None,
        max_call_attempts: int = 5,
        backoff: ExponentialBackoff = ExponentialBackoff(initial=0.1, maximum=10.0, factor=2.0, jitter=0.25),
    ):
        """Initialize the actor client.

        Args:
            resolver: Resolver instance for endpoint discovery
            name: Name of the actor to invoke
            resolve_timeout: Total timeout in seconds for initial worker resolution.
            call_timeout: Timeout in seconds for RPC calls. Defaults to `resolve_timeout`
                when not specified.
            max_call_attempts: Maximum number of RPC call attempts before giving up.
            backoff: Exponential backoff configuration for both resolution and call retries.
        """
        self._resolver = resolver
        self._name = name
        self._resolve_timeout = resolve_timeout
        self._call_timeout = resolve_timeout if call_timeout is None else call_timeout
        self._max_call_attempts = max_call_attempts
        self._backoff = backoff

        self._rpc_client: ActorServiceClientSync | None = None

    def rpc_client(self) -> ActorServiceClientSync:
        """Resolve actor name to an RPC client (single attempt).

        Resolution is attempted once. On failure (empty endpoints or RPC error),
        the exception propagates to the caller. The outer ``call_with_retry`` in
        ``_RpcMethod.__call__`` is responsible for retrying.

        Returns:
            ActorServiceClientSync connected to the resolved endpoint.

        Raises:
            ConnectError(UNAVAILABLE): If no endpoints are found for the actor.
        """
        if self._rpc_client:
            return self._rpc_client

        logger.info("Resolving name %s via %s", self._name, self._resolver)
        result = self._resolver.resolve(self._name)

        if result.is_empty:
            raise ConnectError(
                Code.UNAVAILABLE,
                f"No endpoints found for actor '{self._name}'",
            )

        logger.info(
            "Resolved actor '%s' to %d endpoint(s)",
            self._name,
            len(result.endpoints),
        )
        logger.info("First endpoint: %s", result.first())
        url = result.first().url
        self._rpc_client = ActorServiceClientSync(
            address=url,
            timeout_ms=None if self._call_timeout is None else int(self._call_timeout * 1000),
        )
        return self._rpc_client

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

        def do_call():
            client = self._client.rpc_client()
            resp = client.call(call)
            return unwrap_actor_response(resp)

        def clear_connection(_exc):
            self._client._rpc_client = None

        return call_with_retry(
            f"{self._client._name}.{self._method_name}",
            do_call,
            on_retry=clear_connection,
            max_attempts=self._client._max_call_attempts,
            backoff=self._client._backoff,
        )
