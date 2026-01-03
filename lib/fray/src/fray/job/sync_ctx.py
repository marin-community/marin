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

"""Synchronous execution context for single-threaded computing."""

import threading
from collections.abc import Callable, Generator
from typing import Any, Literal

from fray.job.context import ThreadActorHandle, _ImmediateFuture


class SyncContext:
    """Execution context for synchronous (single-threaded) execution."""

    def __init__(self):
        self._actors: dict[str, Any] = {}
        self._actor_locks: dict[str, threading.Lock] = {}

    def put(self, obj: Any) -> Any:
        """Identity operation - no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, unwrapping _ImmediateFuture if needed."""
        if isinstance(ref, _ImmediateFuture):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> _ImmediateFuture | Generator[_ImmediateFuture, None, None]:
        """Execute function immediately and wrap result."""
        result = fn(*args)
        return _ImmediateFuture(result)

    def wait(self, futures: list[_ImmediateFuture], num_returns: int = 1) -> tuple[list, list]:
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
    ) -> ThreadActorHandle:
        """Create an actor (stateful service) within the synchronous context.

        Args:
            actor_class: The class to instantiate as an actor (not decorated)
            *args: Positional arguments for actor __init__
            name: Optional name for actor discovery/reuse across workers
            get_if_exists: If True and named actor exists, return existing instance
            lifetime: "non_detached" (dies with context) or "detached" (survives job)
            preemptible: Ignored for sync context
            **kwargs: Keyword arguments for actor __init__

        Returns:
            ThreadActorHandle wrapping the actor instance
        """
        if name is not None and name in self._actors:
            if get_if_exists:
                return ThreadActorHandle(self._actors[name], self._actor_locks[name], self)
            else:
                raise ValueError(f"Actor {name} already exists")

        instance = actor_class(*args, **kwargs)
        actor_lock = threading.Lock()

        if name is not None:
            self._actors[name] = instance
            self._actor_locks[name] = actor_lock

        return ThreadActorHandle(instance, actor_lock, self)
