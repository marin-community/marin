"""
Type stubs for fray_rusty - High-performance distributed computing runtime for Fray.

This module provides Rust-based implementations of Fray's core primitives:
- Distributed object storage
- Remote task execution
- Actor model support
- Future-based result handling
"""

from typing import Any, Callable, Optional, TypeVar, overload

T = TypeVar("T")

class RustyFuture:
    """
    A future representing a pending computation result.

    This wraps a TaskRef from the protocol and provides a Python API
    for checking status and retrieving results.
    """

    def __init__(self, task_id: bytes) -> None:
        """
        Create a new RustyFuture.

        Args:
            task_id: The task identifier as bytes.
        """
        ...

    def __repr__(self) -> str:
        """Return a string representation of the future."""
        ...

    def result(self) -> Any:
        """
        Retrieve the result of this future.

        Returns:
            The result object when the task completes.

        Raises:
            RuntimeError: If the task failed or is not yet complete.
        """
        ...

    @property
    def task_id(self) -> bytes:
        """Get the task ID as bytes."""
        ...


class RustyActorMethod:
    """
    A reference to a specific method on a remote actor.

    Created by attribute access on RustyActorHandle. Call .remote() to
    invoke the method asynchronously.
    """

    def __repr__(self) -> str:
        """Return a string representation of the actor method."""
        ...

    def remote(self, *args: Any, **kwargs: Any) -> RustyFuture:
        """
        Invoke this actor method remotely.

        Args:
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            A RustyFuture representing the pending result.
        """
        ...

    @property
    def actor_id(self) -> bytes:
        """Get the actor ID as bytes."""
        ...

    @property
    def method_name(self) -> str:
        """Get the method name."""
        ...


class RustyActorHandle:
    """
    A handle to a remote actor instance.

    Provides access to actor methods via attribute access, which returns
    RustyActorMethod instances that can be invoked with .remote().
    """

    def __init__(self, actor_id: bytes, actor_name: str) -> None:
        """
        Create a new RustyActorHandle.

        Args:
            actor_id: The actor identifier as bytes.
            actor_name: The human-readable name of the actor.
        """
        ...

    def __repr__(self) -> str:
        """Return a string representation of the actor handle."""
        ...

    def __getattr__(self, name: str) -> RustyActorMethod:
        """
        Get an actor method by name.

        Args:
            name: The method name.

        Returns:
            A RustyActorMethod instance that can be invoked with .remote().
        """
        ...

    @property
    def actor_id(self) -> bytes:
        """Get the actor ID as bytes."""
        ...

    @property
    def actor_name(self) -> str:
        """Get the actor name."""
        ...


class RustyContext:
    """
    The main entry point for interacting with the Fray distributed runtime.

    RustyContext manages connections to the coordinator and provides methods
    for storing objects, running tasks, creating actors, and waiting on futures.
    """

    def __init__(self, coordinator_addr: str) -> None:
        """
        Create a new RustyContext.

        Args:
            coordinator_addr: The coordinator address (e.g., "localhost:50051").
        """
        ...

    def __repr__(self) -> str:
        """Return a string representation of the context."""
        ...

    def put(self, obj: Any) -> Any:
        """
        Store an object in the distributed object store.

        Args:
            obj: The Python object to store.

        Returns:
            An ObjectRef that can be used to retrieve the object later.
        """
        ...

    def get(self, object_ref: Any) -> Any:
        """
        Retrieve an object from the distributed object store.

        Args:
            object_ref: The ObjectRef returned by put().

        Returns:
            The deserialized Python object.
        """
        ...

    def run(self, func: Callable[..., T], *args: Any) -> RustyFuture:
        """
        Run a function remotely.

        Args:
            func: The function to execute remotely.
            *args: Positional arguments to pass to the function.

        Returns:
            A RustyFuture representing the pending result.
        """
        ...

    def wait(
        self,
        futures: list[RustyFuture],
        num_returns: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> tuple[list[RustyFuture], list[RustyFuture]]:
        """
        Wait for futures to complete.

        Args:
            futures: A list of RustyFuture instances.
            num_returns: Number of futures to wait for (default: all).
            timeout: Optional timeout in seconds.

        Returns:
            A tuple of (ready, pending) future lists.
        """
        ...

    def create_actor(
        self,
        actor_class: type,
        *args: Any,
        name: Optional[str] = None,
        lifetime: Optional[str] = None,
        **kwargs: Any,
    ) -> RustyActorHandle:
        """
        Create a new actor instance.

        Args:
            actor_class: The actor class to instantiate.
            *args: Positional arguments for the actor constructor.
            name: Optional actor name.
            lifetime: Actor lifetime ('detached' or 'task').
            **kwargs: Keyword arguments for the actor constructor.

        Returns:
            A RustyActorHandle for interacting with the actor.
        """
        ...

    @property
    def coordinator_addr(self) -> str:
        """Get the coordinator address."""
        ...


def init(coordinator: str) -> RustyContext:
    """
    Initialize a new Fray context connected to a coordinator.

    Args:
        coordinator: The coordinator address (e.g., "localhost:50051").

    Returns:
        A RustyContext instance for interacting with the distributed runtime.
    """
    ...
