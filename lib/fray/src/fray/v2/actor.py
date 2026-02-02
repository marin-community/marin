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

"""Actor protocols for fray v2.

Defines the calling convention for remote actors: handle.method.remote()
returns an ActorFuture, handle.method() calls synchronously. ActorGroup
holds a set of actor handles with lifecycle tied to underlying jobs.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Protocol, runtime_checkable

from fray.v2.client import JobHandle
from fray.v2.types import JobStatus


@runtime_checkable
class ActorFuture(Protocol):
    """Future for an actor method call."""

    def result(self, timeout: float | None = None) -> Any:
        """Block until result is available."""
        ...


class FutureActorFuture:
    """ActorFuture backed by a concurrent.futures.Future."""

    def __init__(self, future: Future[Any]):
        self._future = future

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)


@runtime_checkable
class ActorMethod(Protocol):
    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        """Invoke the method remotely. Returns a future."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the method synchronously (blocking)."""
        ...


@runtime_checkable
class ActorHandle(Protocol):
    """Handle to a remote actor with .method.remote() calling convention."""

    def __getattr__(self, method_name: str) -> ActorMethod: ...


class ActorGroup:
    """Group of actor instances with lifecycle tied to underlying jobs.

    Returned immediately from create_actor_group(). For LocalClient all
    actors are ready immediately; remote backends may have actors that
    become available asynchronously.
    """

    def __init__(self, handles: list[ActorHandle], jobs: list[JobHandle]):
        self._handles = handles
        self._jobs = jobs
        self._yielded_count = 0

    @property
    def ready_count(self) -> int:
        """Number of actors that are available for RPC."""
        return len(self._handles)

    def wait_ready(self, count: int | None = None, timeout: float = 300.0) -> list[ActorHandle]:
        """Block until `count` actors are ready (default: all).

        For LocalClient, all actors are immediately ready since they are
        in-process objects. Returns a snapshot of ready handles.
        """
        target = count if count is not None else len(self._handles)
        if target > len(self._handles):
            raise ValueError(f"Requested {target} actors but group only has {len(self._handles)}")
        result = list(self._handles[:target])
        self._yielded_count = max(self._yielded_count, target)
        return result

    @property
    def jobs(self) -> list[JobHandle]:
        """Underlying job handles for lifecycle management."""
        return self._jobs

    def statuses(self) -> list[JobStatus]:
        """Return the job status of each actor in the group."""
        return [job.status() for job in self._jobs]

    def discover_new(self) -> list[ActorHandle]:
        """Return handles that are ready but haven't been yielded yet.

        After wait_ready(count=1), subsequent calls to discover_new() will
        return the remaining handles as they become available. For LocalClient
        all handles are ready immediately, so this returns whatever wait_ready
        didn't return on its first call.
        """
        if self._yielded_count >= len(self._handles):
            return []
        new = list(self._handles[self._yielded_count :])
        self._yielded_count = len(self._handles)
        return new

    def shutdown(self) -> None:
        """Terminate all actor jobs."""
        for job in self._jobs:
            job.terminate()
