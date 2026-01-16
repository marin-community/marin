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
Default timeout (30s) and retry config handle this gracefully.

Example:
    resolver = ClusterResolver("http://controller:8080")
    client = ActorClient(resolver, "my-actor")
    result = client.some_method(arg1, arg2)  # Retries until actor found

Custom retry behavior:
    retry_config = RetryConfig(initial_delay=0.2, max_delay=5.0)
    client = ActorClient(resolver, "my-actor", retry_config=retry_config)
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import cloudpickle

from fluster.actor.types import Resolver, ResolveResult
from fluster.rpc import actor_pb2
from fluster.rpc.actor_connect import ActorServiceClientSync

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry when resolving actors.

    When an actor name cannot be resolved, the client retries with
    exponentially increasing delays until the timeout is reached.

    Attributes:
        initial_delay: Initial retry delay in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        jitter_factor: Random jitter as fraction of delay (e.g., 0.25 = ±25%)
    """

    initial_delay: float = 0.1
    max_delay: float = 2.0
    backoff_factor: float = 2.0
    jitter_factor: float = 0.25

    def __post_init__(self):
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be positive")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")
        if not 0 <= self.jitter_factor < 1.0:
            raise ValueError("jitter_factor must be in [0, 1)")


def calculate_next_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate next retry delay with exponential backoff and jitter.

    Args:
        attempt: Retry attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds before next retry
    """
    # Exponential: initial * (backoff^attempt)
    delay = config.initial_delay * (config.backoff_factor**attempt)

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter: random in [delay*(1-jitter), delay*(1+jitter)]
    if config.jitter_factor > 0:
        jitter_range = delay * config.jitter_factor
        delay = delay + random.uniform(-jitter_range, jitter_range)
        delay = max(0.001, delay)  # Keep positive

    return delay


class ActorClient:
    """Actor client with resolver-based discovery."""

    def __init__(
        self,
        resolver: Resolver,
        name: str,
        timeout: float = 30.0,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize the actor client.

        Args:
            resolver: Resolver instance for endpoint discovery
            name: Name of the actor to invoke
            timeout: Total timeout in seconds for resolution + RPC calls.
                When resolving, retries continue until this timeout is reached.
            retry_config: Retry configuration. If None, uses default RetryConfig().
        """
        self._resolver = resolver
        self._name = name
        self._timeout = timeout
        self._retry_config = retry_config or RetryConfig()
        self._cached_result: ResolveResult | None = None
        self._client: ActorServiceClientSync | None = None
        self._client_url: str | None = None

    def _resolve(self) -> ResolveResult:
        """Resolve actor name with exponential backoff retry.

        Retries resolution with exponential backoff until either:
        - Endpoints are found (success)
        - Timeout is reached (raises TimeoutError)

        Returns:
            ResolveResult with at least one endpoint

        Raises:
            TimeoutError: If no endpoints found within timeout
        """
        # Check cache first
        if self._cached_result is not None and not self._cached_result.is_empty:
            return self._cached_result

        # Retry loop with exponential backoff
        start_time = time.monotonic()
        attempt = 0

        while True:
            # Try to resolve
            result = self._resolver.resolve(self._name)

            if not result.is_empty:
                # Success! Cache and return
                self._cached_result = result
                if attempt > 0:
                    logger.debug(
                        f"Resolved actor '{self._name}' to {len(result.endpoints)} endpoint(s) "
                        f"after {attempt} retries in {time.monotonic() - start_time:.2f}s"
                    )
                return result

            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= self._timeout:
                raise TimeoutError(
                    f"Failed to resolve actor '{self._name}' after {self._timeout}s " f"({attempt} retries)"
                )

            # Calculate next delay with exponential backoff + jitter
            delay = calculate_next_delay(attempt, self._retry_config)

            # Adjust delay to not exceed timeout
            remaining = self._timeout - elapsed
            if delay > remaining:
                delay = remaining

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
