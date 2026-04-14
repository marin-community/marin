# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor client for making RPC calls to actor servers.

The ActorClient provides transparent actor discovery and invocation with
automatic retry logic. Both resolution failures (e.g., actor not yet
registered) and transient RPC errors are retried up to ``max_call_attempts``
with exponential backoff.

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
import time
from typing import Any

import cloudpickle
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.actor.resolver import Resolver
from iris.rpc import actor_pb2
from iris.rpc.actor_connect import ActorServiceClientSync
from iris.rpc.errors import call_with_retry
from rigging.timing import ExponentialBackoff

logger = logging.getLogger(__name__)


def unwrap_actor_response(resp: actor_pb2.ActorResponse) -> Any:
    """Unwrap an ActorResponse, raising the embedded exception on error."""
    if resp.HasField("error"):
        if resp.error.serialized_exception:
            raise cloudpickle.loads(resp.error.serialized_exception)
        raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")
    return cloudpickle.loads(resp.serialized_value)


class ActorClient:
    """Actor client with resolver-based discovery.

    By default the client waits forever, i.e. there's no timeout in httpx.
    Specify ``call_timeout`` to apply a timeout to individual RPC calls.
    """

    def __init__(
        self,
        resolver: Resolver,
        name: str,
        call_timeout: float | None = None,
        max_call_attempts: int = 10,
        backoff: ExponentialBackoff = ExponentialBackoff(initial=0.5, maximum=10.0, factor=2.0, jitter=0.25),
    ):
        """Initialize the actor client.

        Args:
            resolver: Resolver instance for endpoint discovery
            name: Name of the actor to invoke
            call_timeout: Timeout in seconds for individual RPC calls.
                None (default) means no timeout.
            max_call_attempts: Maximum number of RPC call attempts (including
                resolution failures) before giving up.
            backoff: Exponential backoff configuration for retries between attempts.
        """
        self._resolver = resolver
        self._name = name
        self._call_timeout = call_timeout
        self._max_call_attempts = max_call_attempts
        self._backoff = backoff

        self._rpc_client: ActorServiceClientSync | None = None
        self._rpc_headers: dict[str, str] = {}

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
        endpoint = result.first()
        logger.info("First endpoint: url=%s, actor_id=%s", endpoint.url, endpoint.actor_id)
        self._rpc_headers = dict(endpoint.metadata)
        self._rpc_client = ActorServiceClientSync(
            address=endpoint.url,
            timeout_ms=None if self._call_timeout is None else int(self._call_timeout * 1000),
            accept_compression=[],
        )
        return self._rpc_client

    def _clear_connection(self, _exc: Exception) -> None:
        self._rpc_client = None
        self._rpc_headers = {}

    def start_operation(self, method_name: str, *args: Any, **kwargs: Any) -> str:
        """Start a long-running operation. Returns the operation ID."""
        call = actor_pb2.ActorCall(
            method_name=method_name,
            actor_name=self._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        def do_call():
            client = self.rpc_client()
            return client.start_operation(call, headers=self._rpc_headers)

        op = call_with_retry(
            f"{self._name}.start_operation({method_name})",
            do_call,
            on_retry=self._clear_connection,
            max_attempts=self._max_call_attempts,
            backoff=self._backoff,
        )
        return op.operation_id

    def poll_operation_status(self, operation_id: str) -> actor_pb2.Operation:
        """Single-shot poll of a long-running operation's state."""
        req = actor_pb2.OperationId(operation_id=operation_id)

        def do_call():
            return self.rpc_client().get_operation(req, headers=self._rpc_headers)

        return call_with_retry(
            f"{self._name}.poll_operation_status({operation_id[:8]})",
            do_call,
            on_retry=self._clear_connection,
            max_attempts=self._max_call_attempts,
            backoff=self._backoff,
        )

    def get_operation(
        self,
        operation_id: str,
        poll_backoff: ExponentialBackoff | None = None,
    ) -> actor_pb2.Operation:
        """Poll a long-running operation until it completes, using exponential backoff."""
        if poll_backoff is None:
            poll_backoff = ExponentialBackoff(initial=0.1, maximum=10.0, factor=2.0, jitter=0.25)
        while True:
            op = self.poll_operation_status(operation_id)
            if op.state != actor_pb2.Operation.RUNNING:
                return op
            time.sleep(poll_backoff.next_interval())

    def cancel_operation(self, operation_id: str) -> actor_pb2.Operation:
        """Cancel a long-running operation."""
        req = actor_pb2.OperationId(operation_id=operation_id)

        def do_call():
            return self.rpc_client().cancel_operation(req, headers=self._rpc_headers)

        return call_with_retry(
            f"{self._name}.cancel_operation({operation_id[:8]})",
            do_call,
            on_retry=self._clear_connection,
            max_attempts=self._max_call_attempts,
            backoff=self._backoff,
        )

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
            resp = client.call(call, headers=self._client._rpc_headers)
            return unwrap_actor_response(resp)

        return call_with_retry(
            f"{self._client._name}.{self._method_name}",
            do_call,
            on_retry=self._client._clear_connection,
            max_attempts=self._client._max_call_attempts,
            backoff=self._client._backoff,
        )
