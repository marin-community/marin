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

"""Synchronous execution context for single-threaded execution."""

import threading
from collections.abc import Callable
from typing import Any, Literal

__all__ = ["SyncContext"]


class _SyncFuture:
    """Wrapper for immediately available results to match Future interface."""

    def __init__(self, result: Any):
        self._result = result

    def result(self) -> Any:
        return self._result


class _SyncActorHandle:
    """Actor handle for SyncContext - serializes method calls with a lock."""

    def __init__(self, instance: Any, lock: threading.Lock):
        self._instance = instance
        self._lock = lock

    def __getattr__(self, method_name: str):
        method = getattr(self._instance, method_name)
        return _SyncActorMethod(method, self._lock)


class _SyncActorMethod:
    """Method wrapper for SyncContext actors."""

    def __init__(self, method: Callable, lock: threading.Lock):
        self._method = method
        self._lock = lock

    def remote(self, *args, **kwargs) -> _SyncFuture:
        with self._lock:
            result = self._method(*args, **kwargs)
        return _SyncFuture(result)


class SyncContext:
    """Execution context for synchronous (single-threaded) execution."""

    def __init__(self):
        self._actors: dict[str, Any] = {}
        self._actor_locks: dict[str, threading.Lock] = {}

    def put(self, obj: Any) -> Any:
        """Identity operation - no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, unwrapping _SyncFuture if needed."""
        if isinstance(ref, _SyncFuture):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> _SyncFuture:
        """Execute function immediately and wrap result."""
        result = fn(*args)
        return _SyncFuture(result)

    def wait(self, futures: list[_SyncFuture], num_returns: int = 1) -> tuple[list, list]:
        """All futures are immediately ready."""
        return futures[:num_returns], futures[num_returns:]

    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["non_detached", "detached"] = "non_detached",
        preemptible: bool = True,
        **kwargs,
    ) -> _SyncActorHandle:
        if name is not None and name in self._actors:
            if get_if_exists:
                return _SyncActorHandle(self._actors[name], self._actor_locks[name])
            else:
                raise ValueError(f"Actor {name} already exists")

        instance = actor_class(*args, **kwargs)
        actor_lock = threading.Lock()

        if name is not None:
            self._actors[name] = instance
            self._actor_locks[name] = actor_lock

        return _SyncActorHandle(instance, actor_lock)
