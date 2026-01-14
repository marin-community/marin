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

"""Actor pool for load-balanced and broadcast RPC calls."""

import itertools
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import cloudpickle

from fluster import actor_pb2
from fluster.actor.resolver import ResolveResult, ResolvedEndpoint, Resolver
from fluster.actor_connect import ActorServiceClientSync

T = TypeVar("T")


@dataclass
class CallResult:
    """Result of a single call in a broadcast.

    Attributes:
        endpoint: The endpoint that was called
        value: The return value (None if exception occurred)
        exception: The exception raised (None if successful)
    """

    endpoint: ResolvedEndpoint
    value: Any | None = None
    exception: BaseException | None = None

    @property
    def success(self) -> bool:
        """Returns True if the call succeeded without exception."""
        return self.exception is None


class BroadcastFuture(Generic[T]):
    """Future representing results from a broadcast call to multiple endpoints.

    Provides methods to wait for all results, wait for any result, or iterate
    results as they complete.
    """

    def __init__(self, futures: list[tuple[ResolvedEndpoint, Future]]):
        """Initialize with list of (endpoint, future) pairs."""
        self._futures = futures

    def wait_all(self, timeout: float | None = None) -> list[CallResult]:
        """Wait for all calls to complete and return all results.

        Args:
            timeout: Total timeout in seconds for all calls

        Returns:
            List of CallResult, one per endpoint
        """
        results = []
        for endpoint, future in self._futures:
            try:
                value = future.result(timeout=timeout)
                results.append(CallResult(endpoint=endpoint, value=value))
            except Exception as e:
                results.append(CallResult(endpoint=endpoint, exception=e))
        return results

    def wait_any(self, timeout: float | None = None) -> CallResult:
        """Wait for the first call to complete and return its result.

        Args:
            timeout: Timeout in seconds

        Returns:
            CallResult from the first completed call

        Raises:
            TimeoutError: If no results are ready within timeout
        """
        for future in as_completed([f for _, f in self._futures], timeout=timeout):
            idx = next(i for i, (_, f) in enumerate(self._futures) if f is future)
            endpoint = self._futures[idx][0]
            try:
                value = future.result()
                return CallResult(endpoint=endpoint, value=value)
            except Exception as e:
                return CallResult(endpoint=endpoint, exception=e)
        raise TimeoutError("No results within timeout")

    def as_completed(self, timeout: float | None = None) -> Iterator[CallResult]:
        """Iterate over results as they complete.

        Args:
            timeout: Total timeout in seconds for all calls

        Yields:
            CallResult for each completed call
        """
        endpoint_map = {id(f): ep for ep, f in self._futures}
        for future in as_completed([f for _, f in self._futures], timeout=timeout):
            endpoint = endpoint_map[id(future)]
            try:
                value = future.result()
                yield CallResult(endpoint=endpoint, value=value)
            except Exception as e:
                yield CallResult(endpoint=endpoint, exception=e)


class ActorPool(Generic[T]):
    """Pool of actors for load-balanced and broadcast calls.

    Resolves a pool of endpoints for an actor name and provides methods to
    distribute calls across them (round-robin) or broadcast to all endpoints.

    Example:
        >>> pool = ActorPool(resolver, "inference")
        >>> result = pool.call().predict(data)  # Round-robin to one endpoint
        >>> broadcast = pool.broadcast().reload_model()  # Send to all endpoints
        >>> results = broadcast.wait_all()
    """

    def __init__(self, resolver: Resolver, name: str, timeout: float = 30.0):
        """Initialize actor pool.

        Args:
            resolver: Resolver to discover endpoints
            name: Actor name to resolve
            timeout: RPC timeout in seconds
        """
        self._resolver = resolver
        self._name = name
        self._timeout = timeout
        self._round_robin: itertools.cycle | None = None
        self._cached_result: ResolveResult | None = None
        self._executor = ThreadPoolExecutor(max_workers=32)

    def _resolve(self) -> ResolveResult:
        """Resolve endpoints, caching result and updating round-robin iterator."""
        result = self._resolver.resolve(self._name)
        if self._cached_result is None or result.endpoints != self._cached_result.endpoints:
            self._round_robin = itertools.cycle(result.endpoints) if result.endpoints else None
            self._cached_result = result
        return result

    @property
    def size(self) -> int:
        """Number of endpoints in the pool."""
        return len(self._resolve().endpoints)

    @property
    def endpoints(self) -> list[ResolvedEndpoint]:
        """List of resolved endpoints."""
        return list(self._resolve().endpoints)

    def _call_endpoint(
        self,
        endpoint: ResolvedEndpoint,
        method_name: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Make an RPC call to a specific endpoint.

        Args:
            endpoint: Target endpoint
            method_name: Method to call
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Deserialized return value

        Raises:
            Exception from the remote actor method
        """
        client = ActorServiceClientSync(
            address=endpoint.url,
            timeout_ms=int(self._timeout * 1000),
        )

        call = actor_pb2.ActorCall(
            method_name=method_name,
            actor_name=self._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        resp = client.call(call)

        if resp.HasField("error"):
            if resp.error.serialized_exception:
                raise cloudpickle.loads(resp.error.serialized_exception)
            raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

        return cloudpickle.loads(resp.serialized_value)

    def call(self) -> "_PoolCallProxy[T]":
        """Create a proxy for round-robin calls.

        Returns:
            Proxy that distributes method calls across endpoints
        """
        return _PoolCallProxy(self)

    def broadcast(self) -> "_PoolBroadcastProxy[T]":
        """Create a proxy for broadcast calls.

        Returns:
            Proxy that sends method calls to all endpoints
        """
        return _PoolBroadcastProxy(self)


class _PoolCallProxy(Generic[T]):
    """Proxy for round-robin calls to a pool."""

    def __init__(self, pool: ActorPool[T]):
        self._pool = pool

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        """Create a callable that invokes the method on the next endpoint in round-robin."""

        def call(*args, **kwargs):
            self._pool._resolve()
            if self._pool._round_robin is None:
                raise RuntimeError(f"No endpoints for '{self._pool._name}'")
            endpoint = next(self._pool._round_robin)
            return self._pool._call_endpoint(endpoint, method_name, args, kwargs)

        return call


class _PoolBroadcastProxy(Generic[T]):
    """Proxy for broadcast calls to all endpoints in a pool."""

    def __init__(self, pool: ActorPool[T]):
        self._pool = pool

    def __getattr__(self, method_name: str) -> Callable[..., BroadcastFuture]:
        """Create a callable that invokes the method on all endpoints in parallel."""

        def broadcast(*args, **kwargs) -> BroadcastFuture:
            result = self._pool._resolve()
            futures = []
            for endpoint in result.endpoints:
                future = self._pool._executor.submit(
                    self._pool._call_endpoint,
                    endpoint,
                    method_name,
                    args,
                    kwargs,
                )
                futures.append((endpoint, future))
            return BroadcastFuture(futures)

        return broadcast
