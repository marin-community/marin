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

"""Monarch-backed JobContext implementation for Fray."""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable
from typing import Any

from fray.job import JobContext
from fray.types import ActorOptions, TaskOptions

from .monarch_helpers import (
    MONARCH_AVAILABLE,
    MonarchActorHandle,
    MonarchObjectRef,
    ObjectStoreActor,
    generate_object_id,
)

if MONARCH_AVAILABLE:
    from monarch.actor import Actor, ProcMesh, endpoint
    from monarch._src.actor.host_mesh import create_local_host_mesh
else:
    # Stubs for type checking
    Actor = object
    ProcMesh = object
    create_local_host_mesh = None

    def endpoint(fn):
        return fn


# Context variable for propagating JobContext to nested tasks/actors
_job_context: contextvars.ContextVar[MonarchJobContext] = contextvars.ContextVar("job_context")


# Task actor class defined at module level to avoid closure captures
if MONARCH_AVAILABLE:

    class _TaskActor(Actor):
        """Ephemeral actor that executes a single task."""

        def __init__(self, procs: ProcMesh, object_store: Any, fn: Callable, args: tuple, kwargs: dict):
            super().__init__()
            self._procs = procs
            self._object_store = object_store
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        @endpoint
        def run(self):
            # Reconstruct context in actor for nested operations
            ctx = MonarchJobContext.__new__(MonarchJobContext)
            ctx._procs = self._procs
            ctx._object_store = self._object_store
            ctx._resource_config = {}
            ctx._object_counter = 0
            ctx._task_counter = 0
            ctx._actor_counter = 0

            # Set as current context for nested calls
            _job_context.set(ctx)

            # Execute user function
            return self._fn(*self._args, **self._kwargs)

else:
    _TaskActor = None


class MonarchJobContext(JobContext):
    """Monarch-backed implementation of Fray JobContext.

    This implementation maps Fray's task and actor APIs to Monarch's actor-based
    execution model. Key design decisions:

    1. Tasks are implemented as single-use actors with @endpoint methods
    2. Actors map directly to Monarch actor meshes
    3. Object store is implemented using a dedicated ObjectStoreActor
    4. Resources are specified at process spawn time (process-level, not per-task)
    5. Context propagation uses contextvars to maintain JobContext across actors
    """

    def __init__(self, resource_config: dict[str, Any] | None = None, procs: ProcMesh | None = None):
        """
        Initialize MonarchJobContext.

        Args:
            resource_config: Dict specifying resources like {"gpus": 8}.
                           Used to spawn worker processes automatically via create_local_host_mesh().
            procs: Pre-spawned Monarch ProcMesh object. If provided, resource_config is ignored.

        Raises:
            ImportError: If Monarch is not available
        """
        if not MONARCH_AVAILABLE:
            raise ImportError("Monarch is not available. Install torchmonarch-nightly on a supported platform.")

        # Auto-spawn processes if not provided
        if procs is None:
            if resource_config is None:
                # Default: spawn with minimal resources (1 CPU process)
                resource_config = {}

            # Create local host mesh first (correct pattern for initialization from regular Python)
            host = create_local_host_mesh()
            # Spawn processes on the host
            procs = host.spawn_procs(name="_fray_workers", per_host=resource_config)

        self._procs = procs
        self._resource_config = resource_config or {}

        # Initialize object store actor for put/get functionality
        self._object_store = self._procs.spawn("_fray_object_store", ObjectStoreActor)

        # Counters for generating unique names
        self._object_counter = 0
        self._task_counter = 0
        self._actor_counter = 0

    def create_task(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        options: TaskOptions | None = None,
    ) -> Any:
        """
        Create a task by wrapping function in a single-use actor.

        Since Monarch uses actors instead of tasks, we create a task actor
        that wraps the user's function in an @endpoint method.

        Args:
            fn: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            options: Task options (resources, runtime_env, etc.)

        Returns:
            MonarchObjectRef wrapping the Monarch Future
        """
        if kwargs is None:
            kwargs = {}

        # Generate unique task ID
        task_id = self._task_counter
        self._task_counter += 1

        # Spawn task actor with function and arguments
        actor_name = f"_fray_task_{task_id}"
        task_actor = self._procs.spawn(actor_name, _TaskActor, self._procs, self._object_store, fn, args, kwargs)

        # Call the run endpoint
        future = task_actor.run.call()

        # Return ref that extracts result
        return MonarchObjectRef(future, take_first=False)

    def create_actor(
        self,
        klass: type,
        args: tuple = (),
        kwargs: dict | None = None,
        options: ActorOptions | None = None,
    ) -> Any:
        """
        Create a long-lived actor.

        Wraps the user's actor class to inject Fray context, then spawns a Monarch
        actor mesh. The mesh is wrapped in MonarchActorHandle to provide single-actor
        semantics (one actor instance rather than a mesh).

        Args:
            klass: Actor class to instantiate
            args: Positional arguments for actor __init__
            kwargs: Keyword arguments for actor __init__
            options: Actor options (name, resources, lifetime, etc.)

        Returns:
            MonarchActorHandle wrapping the Monarch Mesh
        """
        if kwargs is None:
            kwargs = {}

        # Capture picklable parts of context
        procs = self._procs
        object_store = self._object_store

        # Wrap actor class to inject Fray context
        class ContextualActor(klass):
            def __init__(self, *init_args, **init_kwargs):
                super().__init__(*init_args, **init_kwargs)
                # Store picklable references for context reconstruction
                self._fray_procs = procs
                self._fray_object_store = object_store

                # Reconstruct context for nested operations
                ctx = MonarchJobContext.__new__(MonarchJobContext)
                ctx._procs = self._fray_procs
                ctx._object_store = self._fray_object_store
                ctx._resource_config = {}
                ctx._object_counter = 0
                ctx._task_counter = 0
                ctx._actor_counter = 0

                # Set as current context
                _job_context.set(ctx)

        # Determine actor name
        actor_name = None
        if options and options.name:
            actor_name = options.name
            # Check if we should reuse existing actor
            if options.get_if_exists:
                # TODO: Implement actor lookup by name
                # For now, we'll create a new actor
                pass
        else:
            # Generate unique actor name
            actor_id = self._actor_counter
            self._actor_counter += 1
            actor_name = f"{klass.__name__}_{actor_id}"

        # Spawn actor instance
        actor = self._procs.spawn(actor_name, ContextualActor, args=args, kwargs=kwargs)

        # Wrap actor to provide Fray-compatible API
        return MonarchActorHandle(actor)

    def get(self, ref: Any) -> Any:
        """
        Block and retrieve result from object reference.

        Handles both task/actor results (MonarchObjectRef wrapping Futures) and
        object store references.

        Args:
            ref: MonarchObjectRef to resolve

        Returns:
            The result value

        Raises:
            TypeError: If ref is not a MonarchObjectRef
        """
        if not isinstance(ref, MonarchObjectRef):
            raise TypeError(f"Expected MonarchObjectRef, got {type(ref)}")

        # Handle object store references
        if ref._is_object_store_ref:
            # Retrieve from object store
            future = self._object_store.get.call(ref._obj_id)
            results = future.get()
            # Object store is a single-instance actor, return first result
            if isinstance(results, list) and len(results) > 0:
                return results[0]
            return results

        # Handle task/actor future references
        results = ref._future.get()

        # If this is a single-task reference, return first result
        if ref._take_first:
            if isinstance(results, list) and len(results) > 0:
                return results[0]
            return results

        # Otherwise return all results (mesh semantics)
        return results

    def wait(
        self,
        refs: list[Any],
        num_returns: int = 1,
        timeout: float | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """
        Wait for some references to complete.

        Monarch futures don't have built-in wait_any/wait_all semantics,
        so we implement polling-based wait.

        Args:
            refs: List of MonarchObjectRef to wait on
            num_returns: Number of results to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Tuple of (ready_refs, not_ready_refs)
        """
        ready = []
        not_ready = list(refs)
        start_time = time.time()

        while len(ready) < num_returns and len(not_ready) > 0:
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                break

            # Poll futures by attempting to get with very short operations
            # Note: Monarch Future.get() blocks, so we can't truly poll
            # This is a limitation - we'll check them in order
            for ref in not_ready[:]:
                try:
                    # For now, we'll use a blocking approach
                    # TODO: Implement actual polling if Monarch supports it
                    if len(ready) >= num_returns:
                        break

                    # Try to get the result (this will block until ready)
                    # This is not ideal but matches Monarch's API
                    _ = self.get(ref)
                    ready.append(ref)
                    not_ready.remove(ref)
                except Exception:
                    # Future not ready or error occurred
                    continue

            # Small sleep to avoid busy waiting
            if len(ready) < num_returns and len(not_ready) > 0:
                time.sleep(0.1)

        return ready, not_ready

    def put(self, obj: Any) -> Any:
        """
        Store object in distributed object store.

        Uses a dedicated ObjectStoreActor to implement object store functionality,
        since Monarch doesn't have a built-in object store like Ray.

        Args:
            obj: Object to store

        Returns:
            MonarchObjectRef referencing the stored object
        """
        # Generate unique object ID
        obj_id = generate_object_id()

        # Store in object store actor
        future = self._object_store.put.call(obj_id, obj)

        # Wait for storage to complete
        future.get()

        # Return reference with object ID
        return MonarchObjectRef(obj_id=obj_id)


def get_job_context() -> MonarchJobContext:
    """
    Get the current JobContext from context variable.

    Returns:
        The current MonarchJobContext

    Raises:
        LookupError: If no context is set
    """
    return _job_context.get()


def set_job_context(ctx: MonarchJobContext) -> None:
    """
    Set the current JobContext in context variable.

    Args:
        ctx: MonarchJobContext to set as current
    """
    _job_context.set(ctx)
