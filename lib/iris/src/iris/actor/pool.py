# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor pool for load-balanced and broadcast RPC calls."""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import cloudpickle

from iris.actor.client import unwrap_actor_response
from iris.actor.resolver import ResolvedEndpoint, ResolveResult, Resolver
from iris.rpc import actor_pb2
from iris.rpc.actor_connect import ActorServiceClientSync
from iris.rpc.errors import call_with_retry
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)

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

    def __init__(
        self,
        resolver: Resolver,
        name: str,
        timeout: float = 30.0,
        max_call_attempts: int = 5,
        backoff: ExponentialBackoff = ExponentialBackoff(initial=0.1, maximum=10.0, factor=2.0, jitter=0.25),
        resolve_ttl: float = 5.0,
    ):
        """Initialize actor pool.

        Args:
            resolver: Resolver to discover endpoints
            name: Actor name to resolve
            timeout: RPC timeout in seconds
            max_call_attempts: Maximum number of RPC call attempts before giving up.
            backoff: Exponential backoff configuration for call retries.
            resolve_ttl: Seconds to cache resolve results before re-querying the resolver
        """
        self._resolver = resolver
        self._name = name
        self._timeout = timeout
        self._max_call_attempts = max_call_attempts
        self._backoff = backoff
        self._resolve_ttl = resolve_ttl
        self._endpoint_index = 0
        self._cached_result: ResolveResult | None = None
        self._last_resolve_time: float = 0.0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=32)
        self._clients: dict[str, ActorServiceClientSync] = {}

    def _get_client(self, endpoint: ResolvedEndpoint) -> ActorServiceClientSync:
        """Return a cached client for the endpoint, creating one if needed."""
        url = endpoint.url
        with self._lock:
            client = self._clients.get(url)
            if client is not None:
                return client
            client = ActorServiceClientSync(
                address=url,
                timeout_ms=int(self._timeout * 1000),
                accept_compression=[],
            )
            self._clients[url] = client
            return client

    def _resolve(self) -> ResolveResult:
        now = time.monotonic()
        with self._lock:
            if self._cached_result is not None and (now - self._last_resolve_time) < self._resolve_ttl:
                return self._cached_result

        result = self._resolver.resolve(self._name)
        if result.endpoints:
            with self._lock:
                self._cached_result = result
                self._last_resolve_time = time.monotonic()
        return result

    def _invalidate_resolve_cache(self) -> None:
        """Force the next _resolve() call to re-query the resolver."""
        with self._lock:
            self._last_resolve_time = 0.0
            self._cached_result = None

    def _evict_client(self, url: str) -> None:
        """Remove and close a cached client so it is recreated on next use."""
        with self._lock:
            client = self._clients.pop(url, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Error closing evicted client for %s", url, exc_info=True)

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
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                logger.debug("Error closing client during shutdown", exc_info=True)

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
        client = self._get_client(endpoint)

        call = actor_pb2.ActorCall(
            method_name=method_name,
            actor_name=self._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        resp = client.call(call)
        return unwrap_actor_response(resp)

    def call(self) -> "_PoolCallProxy[T]":
        return _PoolCallProxy(self)

    def broadcast(self) -> "_PoolBroadcastProxy[T]":
        return _PoolBroadcastProxy(self)


class _PoolCallProxy(Generic[T]):
    def __init__(self, pool: ActorPool[T]):
        self._pool = pool

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        def call(*args, **kwargs):
            last_url: list[str | None] = [None]

            def do_call():
                endpoint = self._pool._get_next_endpoint()
                last_url[0] = endpoint.url
                return self._pool._call_endpoint(endpoint, method_name, args, kwargs)

            def on_retry(_exc):
                self._pool._invalidate_resolve_cache()
                if last_url[0] is not None:
                    self._pool._evict_client(last_url[0])

            return call_with_retry(
                f"{self._pool._name}.{method_name}",
                do_call,
                on_retry=on_retry,
                max_attempts=self._pool._max_call_attempts,
                backoff=self._pool._backoff,
            )

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
