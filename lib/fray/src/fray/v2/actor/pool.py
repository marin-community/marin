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

"""Actor pool implementations for Fray v2."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, runtime_checkable

import cloudpickle
import httpx


@runtime_checkable
class BroadcastResult(Protocol):
    """Protocol for broadcast results."""

    def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Wait for all results."""
        ...

    def wait_any(self, timeout: float | None = None) -> Any:
        """Wait for first result."""
        ...


@runtime_checkable
class ActorPool(Protocol):
    """Protocol for actor pools."""

    @property
    def size(self) -> int:
        """Current number of actors in the pool."""
        ...

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None:
        """Wait until pool has at least min_size actors."""
        ...

    def call(self) -> Any:
        """Get proxy for round-robin calls."""
        ...

    def broadcast(self) -> Any:
        """Get proxy for broadcasting to all actors."""
        ...


class FixedActorCallProxy:
    """Proxy for calling actors via HTTP (for FixedResolver)."""

    def __init__(self, addresses: list[str], call_index: int):
        self._addresses = addresses
        self._call_index = call_index

    def __getattr__(self, method_name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> Any:
            if not self._addresses:
                raise RuntimeError("No actors available")

            address = self._addresses[self._call_index % len(self._addresses)]
            url = f"http://{address}" if not address.startswith("http") else address

            # Serialize call
            payload = cloudpickle.dumps(
                {
                    "method": method_name,
                    "args": args,
                    "kwargs": kwargs,
                }
            )

            # Make HTTP call
            response = httpx.post(f"{url}/call", content=payload, timeout=30.0)
            response.raise_for_status()

            # Deserialize result
            result_data = cloudpickle.loads(response.content)
            if "error" in result_data:
                raise RuntimeError(result_data["error"])
            return result_data["result"]

        return method


class FixedBroadcastResult:
    """Broadcast result for fixed addresses."""

    def __init__(self, results: list[Any], errors: list[Exception | None]):
        self._results = results
        self._errors = errors

    def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Return all results (may include exceptions)."""
        return [e if e is not None else r for r, e in zip(self._results, self._errors, strict=True)]

    def wait_any(self, timeout: float | None = None) -> Any:
        """Return first successful result."""
        for result, error in zip(self._results, self._errors, strict=True):
            if error is None:
                return result
        if self._errors:
            raise self._errors[0]  # type: ignore[misc]
        raise RuntimeError("No results")


class FixedBroadcastProxy:
    """Proxy for broadcasting to fixed addresses."""

    def __init__(self, addresses: list[str]):
        self._addresses = addresses

    def __getattr__(self, method_name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> FixedBroadcastResult:
            results = []
            errors: list[Exception | None] = []

            with ThreadPoolExecutor(max_workers=len(self._addresses)) as executor:

                def call_one(address: str) -> tuple[Any, Exception | None]:
                    try:
                        proxy = FixedActorCallProxy([address], 0)
                        result = getattr(proxy, method_name)(*args, **kwargs)
                        return result, None
                    except Exception as e:
                        return None, e

                futures = [executor.submit(call_one, addr) for addr in self._addresses]
                for future in futures:
                    result, error = future.result()
                    results.append(result)
                    errors.append(error)

            return FixedBroadcastResult(results, errors)

        return method


class FixedActorPool:
    """Actor pool with fixed addresses (for FixedResolver)."""

    def __init__(self, name: str, addresses: list[str]):
        self._name = name
        self._addresses = list(addresses)
        self._call_index = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._addresses)

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None:
        """Wait until pool has at least min_size actors."""
        start = time.monotonic()
        while self.size < min_size:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Pool did not reach size {min_size} in {timeout}s")
            time.sleep(0.1)

    def call(self) -> FixedActorCallProxy:
        """Get proxy for round-robin calls."""
        with self._lock:
            index = self._call_index
            self._call_index += 1
        return FixedActorCallProxy(self._addresses, index)

    def broadcast(self) -> FixedBroadcastProxy:
        """Get proxy for broadcasting to all actors."""
        return FixedBroadcastProxy(self._addresses)
