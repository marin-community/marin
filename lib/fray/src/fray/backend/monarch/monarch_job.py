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

import time
from collections.abc import Callable
from typing import Any

from monarch.actor import Actor, ActorError, ProcMesh, endpoint, this_proc

from fray.context import set_job_context
from fray.job import JobContext
from fray.types import ActorOptions, TaskOptions

from .monarch_object import (
    MonarchActorHandle,
    MonarchObjectRef,
    ObjectStoreActor,
    generate_object_id,
)


class _TaskWrapper:
    """Ephemeral wrapper that executes a single task."""

    def __init__(self, fn: Callable, args: tuple, kwargs: dict):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        return self._fn(*self._args, **self._kwargs)


def _create_monarch_actor_class(
    actor_class: type,
    object_store: ObjectStoreActor,
    resource_config: dict[str, Any],
    args: tuple,
    kwargs: dict,
) -> type:
    """Factory function to create a Monarch actor class at runtime with endpoint methods."""

    def __init__(self):
        super(_MonarchActor, self).__init__()
        self._job_context = MonarchJobContext(
            resource_config=resource_config, procs=this_proc(), object_store=object_store
        )
        set_job_context(self._job_context)
        self._user_actor = actor_class(*args, **kwargs)

    # Build dict of methods to add to the class
    class_dict = {"__init__": __init__}

    # Inspect the actor_class directly to find methods
    for attr_name in dir(actor_class):
        if attr_name.startswith("_"):
            continue
        attr = getattr(actor_class, attr_name)
        if callable(attr):

            def make_method(method_name):
                @endpoint
                def endpoint_method(self, *args, **kwargs):
                    set_job_context(self._job_context)
                    return getattr(self._user_actor, method_name)(*args, **kwargs)

                return endpoint_method

            class_dict[attr_name] = make_method(attr_name)

    _MonarchActor = type("_MonarchActor", (Actor,), class_dict)
    return _MonarchActor


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

    _procs: ProcMesh
    _resource_config: dict[str, Any]
    _object_store: ObjectStoreActor

    def __init__(
        self,
        resource_config: dict[str, Any] | None = None,
        procs: ProcMesh | None = None,
        object_store: ObjectStoreActor | None = None,
    ):
        """
        Initialize MonarchJobContext.

        Args:
            resource_config: Dict specifying resources like {"gpus": 8}.
                            Used to spawn worker processes automatically via create_local_host_mesh().
            procs: Pre-spawned Monarch ProcMesh object. If provided, resource_config is ignored.

        Raises:
            ImportError: If Monarch is not available
        """
        self._resource_config = resource_config or {}

        if procs is None:
            host = this_proc().host_mesh
            procs = host.spawn_procs(name="_fray_workers", per_host=self._resource_config)
        self._procs = procs

        if object_store is not None:
            self._object_store = object_store
        else:
            self._object_store = self._procs.spawn("_fray_object_store", ObjectStoreActor)

        self._task_counter = 0
        self._actor_counter = 0

        # Named actor registry for get_if_exists support
        self._named_actors: dict[str, MonarchActorHandle] = {}

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

        # Create a task wrapper and use create_actor to spawn it
        task_actor = self.create_actor(_TaskWrapper, args=(fn, args, kwargs))

        # Call the run method
        return task_actor.run()

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

        # Check if named actor already exists and get_if_exists is True
        if options and options.name and options.get_if_exists:
            existing_actor = self._named_actors.get(options.name)
            if existing_actor is not None:
                return existing_actor

        actor_id = self._actor_counter
        self._actor_counter += 1
        actor_name = f"{klass.__name__}_{actor_id}"

        # Create the actor class dynamically
        actor_class = _create_monarch_actor_class(
            actor_class=klass,
            object_store=self._object_store,
            resource_config=self._resource_config,
            args=args,
            kwargs=kwargs,
        )

        actor = self._procs.spawn(actor_name, actor_class)

        # Wrap actor to provide Fray-compatible API
        handle = MonarchActorHandle(actor)

        # Register named actor for future get_if_exists calls
        if options and options.name:
            self._named_actors[options.name] = handle

        return handle

    def get(self, ref: Any, timeout: float | None = None) -> Any:
        """
        Block and retrieve result from object reference.

        Handles both task/actor results (MonarchObjectRef wrapping Futures) and
        object store references. Can also handle lists of refs.

        Args:
            ref: MonarchObjectRef to resolve, or list of MonarchObjectRef

        Returns:
            The result value, or list of results if ref was a list

        Raises:
            TypeError: If ref is not a MonarchObjectRef or list of MonarchObjectRef
        """
        # Handle list of refs
        if isinstance(ref, list):
            return [self.get(r) for r in ref]

        if not isinstance(ref, MonarchObjectRef):
            raise TypeError(f"Expected MonarchObjectRef, got {type(ref)}")

        # Handle object store references
        if ref._is_object_store_ref:
            # Retrieve from object store
            future = self._object_store.get.call(ref._obj_id)
            try:
                results = future.get(timeout=timeout)
            except ActorError as e:
                # Unwrap and re-raise the original exception
                if e.__cause__:
                    raise e.__cause__ from None
                raise

            # Object store returns ValueMesh results like task/actor calls
            # Need to unwrap the mesh structure
            if hasattr(results, "__iter__") and not isinstance(results, (str | bytes)):
                # Try to extract from mesh results
                results_list = list(results)
                if len(results_list) > 0:
                    # Each element is (mesh_coords, value)
                    if isinstance(results_list[0], tuple) and len(results_list[0]) == 2:
                        # Extract just the values, ignoring mesh coordinates
                        values = [item[1] for item in results_list]
                        # For single-value results, return the value directly
                        if len(values) == 1:
                            return values[0]
                        return values

            # Fallback for non-mesh results
            return results

        # Handle task/actor future references
        try:
            results = ref._future.get(timeout=timeout)
        except ActorError as e:
            # Unwrap and re-raise the original exception
            if e.__cause__:
                raise e.__cause__ from None
            raise

        # Monarch returns a ValueMesh - need to extract actual values
        # ValueMesh typically looks like: (({}, value),) for single-process meshes
        # or a list of (coords, value) tuples for multi-process meshes
        if hasattr(results, "__iter__") and not isinstance(results, (str | bytes)):
            # Try to extract from mesh results
            results_list = list(results)
            if len(results_list) > 0:
                # Each element is (mesh_coords, value)
                if isinstance(results_list[0], tuple) and len(results_list[0]) == 2:
                    # Extract just the values, ignoring mesh coordinates
                    values = [item[1] for item in results_list]
                    # For single-value results, return the value directly
                    if len(values) == 1:
                        return values[0]
                    return values

        # Fallback for non-mesh results
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
        deadline = time.time() + timeout if timeout is not None else None

        while len(ready) < num_returns and len(not_ready) > 0 and time.time() < (deadline or float("inf")):
            for ref in not_ready[:]:
                if len(ready) >= num_returns:
                    break
                try:
                    _ = self.get(ref, timeout=max(0.001, deadline - time.time()) if deadline else None)
                    ready.append(ref)
                    not_ready.remove(ref)
                except TimeoutError:
                    pass

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
