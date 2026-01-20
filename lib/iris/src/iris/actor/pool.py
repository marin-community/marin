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

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import cloudpickle

from iris.actor.resolver import ResolvedEndpoint, ResolveResult, Resolver
from iris.rpc import actor_pb2
from iris.rpc.actor_connect import ActorServiceClientSync

T = TypeVar("T")


@dataclass
class CallResult:
    """Result of a single call in a broadcast."""

    endpoint: ResolvedEndpoint
    value: Any | None = None
    exception: BaseException | None = None

    @property
    def success(self) -> bool:
        return self.exception is None


class BroadcastFuture(Generic[T]):
    """Future representing results from a broadcast call to multiple endpoints."""

    def __init__(self, futures: list[tuple[ResolvedEndpoint, Future]]):
        self._futures = futures

    def wait_all(self, timeout: float | None = None) -> list[CallResult]:
        results = []
        for endpoint, future in self._futures:
            try:
                value = future.result(timeout=timeout)
                results.append(CallResult(endpoint=endpoint, value=value))
            except Exception as e:
                results.append(CallResult(endpoint=endpoint, exception=e))
        return results

    def wait_any(self, timeout: float | None = None) -> CallResult:
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
        """Iterate over results as they complete."""
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
        self._endpoint_index = 0
        self._cached_result: ResolveResult | None = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=32)

    def _resolve(self) -> ResolveResult:
        result = self._resolver.resolve(self._name)
        with self._lock:
            self._cached_result = result
        return result

    def _get_next_endpoint(self) -> ResolvedEndpoint:
        """Get the next endpoint in round-robin order.

        Thread-safe: uses a lock to protect the endpoint index.
        """
        endpoints = self._resolve().endpoints
        with self._lock:
            if not endpoints:
                raise RuntimeError(f"No endpoints for '{self._name}'")
            endpoint = endpoints[self._endpoint_index % len(endpoints)]
            self._endpoint_index += 1
            return endpoint

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "ActorPool[T]":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()

    @property
    def size(self) -> int:
        return len(self._resolve().endpoints)

    @property
    def endpoints(self) -> list[ResolvedEndpoint]:
        return list(self._resolve().endpoints)

    def _call_endpoint(
        self,
        endpoint: ResolvedEndpoint,
        method_name: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
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
        return _PoolCallProxy(self)

    def broadcast(self) -> "_PoolBroadcastProxy[T]":
        return _PoolBroadcastProxy(self)


class _PoolCallProxy(Generic[T]):
    def __init__(self, pool: ActorPool[T]):
        self._pool = pool

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        def call(*args, **kwargs):
            endpoint = self._pool._get_next_endpoint()
            return self._pool._call_endpoint(endpoint, method_name, args, kwargs)

        return call


class _PoolBroadcastProxy(Generic[T]):
    def __init__(self, pool: ActorPool[T]):
        self._pool = pool

    def __getattr__(self, method_name: str) -> Callable[..., BroadcastFuture]:
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
