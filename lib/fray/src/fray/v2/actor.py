# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor protocols for fray v2.

Defines the calling convention for remote actors: handle.method.remote()
returns an ActorFuture, handle.method() calls synchronously. ActorGroup
holds a set of actor handles with lifecycle tied to underlying jobs.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol


class ActorHandle(Protocol):
    """Handle to a remote actor with .method.remote() calling convention."""

    def __getattr__(self, method_name: str) -> ActorMethod: ...


@dataclass(frozen=True)
class ActorContext:
    """Context available to actors during execution."""

    handle: ActorHandle
    """Handle to self, can be passed to other actors for callbacks."""

    index: int
    """The actor's index within its group (0 to count-1)."""

    group_name: str
    """The name of the actor group this actor belongs to."""


_current_actor_ctx: ContextVar[ActorContext | None] = ContextVar("actor_context", default=None)


def current_actor() -> ActorContext:
    """Get the current actor's context. Must be called from within an actor.

    Returns the actor's handle (for passing to other actors), index, and group name.

    Raises:
        RuntimeError: If called outside of an actor context.
    """
    ctx = _current_actor_ctx.get()
    if ctx is None:
        raise RuntimeError("current_actor() called outside of an actor context")
    return ctx


def _set_current_actor(ctx: ActorContext):
    """Set the current actor context. Used by backends during actor creation."""
    return _current_actor_ctx.set(ctx)


def _reset_current_actor(token):
    """Reset the current actor context. Used by backends after actor creation."""
    _current_actor_ctx.reset(token)


class ActorFuture(Protocol):
    """Future for an actor method call."""

    def result(self, timeout: float | None = None) -> Any:
        """Block until result is available."""
        ...


class ActorMethod(Protocol):
    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        """Invoke the method remotely. Returns a future."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the method synchronously (blocking)."""
        ...


class ActorGroup(Protocol):
    """Group of actor instances with lifecycle tied to underlying jobs.

    Returned immediately from create_actor_group(). For LocalClient all
    actors are ready immediately; remote backends may have actors that
    become available asynchronously.
    """

    @property
    def ready_count(self) -> int:
        """Number of actors that are available for RPC."""
        ...

    def wait_ready(self, count: int | None = None, timeout: float = 300.0) -> list[ActorHandle]: ...

    def discover_new(self) -> list[ActorHandle]:
        """Return handles that are ready but haven't been yielded yet.

        After wait_ready(count=1), subsequent calls to discover_new() will
        return the remaining handles as they become available. For LocalClient
        all handles are ready immediately, so this returns whatever wait_ready
        didn't return on its first call.
        """
        ...

    def shutdown(self) -> None:
        """Terminate all actor jobs."""
        ...
