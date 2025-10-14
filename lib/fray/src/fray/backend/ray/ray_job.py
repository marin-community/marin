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

"""Ray backend implementation for JobContext."""

from collections.abc import Callable
from typing import Any

import ray

from fray.job import JobContext
from fray.types import Lifetime


class _RayMethodWrapper:
    """
    Wrapper for Ray actor methods that automatically adds .remote().

    This provides a uniform API between Ray and in-memory backends,
    allowing actor methods to be called directly without needing .remote().
    """

    def __init__(self, method):
        self._method = method

    def __call__(self, *args, **kwargs):
        return self._method.remote(*args, **kwargs)


class RayActorHandle:
    """
    Wrapper for Ray ActorHandle that provides uniform API across backends.

    Automatically appends .remote() to method calls, matching the in-memory
    backend's API where actor methods are called directly.
    """

    def __init__(self, actor_handle):
        self._actor = actor_handle

    def __getattr__(self, name):
        attr = getattr(self._actor, name)
        if callable(attr):
            return _RayMethodWrapper(attr)
        return attr


class RayJobContext(JobContext):
    """Ray-based job context implementation."""

    def create_task(self, fn: Callable, args: tuple = (), kwargs: dict | None = None, options: Any | None = None) -> Any:
        """
        Schedule a task to run asynchronously using Ray.

        Supports resource constraints, per-task runtime environments, and task naming.

        Args:
            fn: Function to execute as a task
            args: Positional arguments to pass to fn
            kwargs: Keyword arguments to pass to fn
            options: Task creation options (TaskOptions: resources, runtime_env, name)

        Returns:
            ObjectRef: Ray ObjectRef for the future result
        """
        if kwargs is None:
            kwargs = {}

        # Capture current context to propagate to task
        current_ctx = self

        # Create wrapper that sets context before running fn
        def task_with_context(*task_args, **task_kwargs):
            from fray.context import set_job_context

            set_job_context(current_ctx)
            return fn(*task_args, **task_kwargs)

        # Wrap the function as a Ray remote task
        remote_fn = ray.remote(task_with_context)

        # Apply options if provided
        if options is not None:
            ray_options = {}

            # Translate resources
            if options.resources:
                # Extract CPU, GPU, and memory to Ray-specific parameters
                cpu = options.resources.get("CPU")
                gpu = options.resources.get("GPU")
                memory = options.resources.get("memory")

                if cpu is not None:
                    ray_options["num_cpus"] = cpu
                if gpu is not None:
                    ray_options["num_gpus"] = gpu
                if memory is not None:
                    ray_options["memory"] = memory

                # Pass other custom resources through to resources dict
                other_resources = {k: v for k, v in options.resources.items() if k not in ("CPU", "GPU", "memory")}
                if other_resources:
                    ray_options["resources"] = other_resources

            # Translate runtime_env
            if options.runtime_env:
                ray_runtime_env = {}

                if options.runtime_env.package_requirements:
                    ray_runtime_env["pip"] = options.runtime_env.package_requirements

                if options.runtime_env.env:
                    ray_runtime_env["env_vars"] = options.runtime_env.env

                # Note: minimum_resources and maximum_resources are job-level concerns
                # and are ignored for per-task runtime_env

                if ray_runtime_env:
                    ray_options["runtime_env"] = ray_runtime_env

            # Task naming for observability
            if options.name:
                ray_options["name"] = options.name

            # Limit task worker reuse (useful for TPU cleanup)
            if options.max_calls is not None:
                ray_options["max_calls"] = options.max_calls

            if ray_options:
                remote_fn = remote_fn.options(**ray_options)

        return remote_fn.remote(*args, **kwargs)

    def get(self, ref: Any) -> Any:
        """
        Block and retrieve result from a Ray ObjectRef.

        Args:
            ref: Single ObjectRef or list of ObjectRefs

        Returns:
            The computed value(s)
        """
        return ray.get(ref)

    def wait(self, refs: list[Any], num_returns: int = 1, timeout: float | None = None) -> tuple[list[Any], list[Any]]:
        """
        Wait for Ray tasks to complete.

        Args:
            refs: List of ObjectRefs to wait on
            num_returns: Number of tasks to wait for before returning
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            Tuple of (done_refs, not_done_refs)
        """
        return ray.wait(refs, num_returns=num_returns, timeout=timeout)

    def put(self, obj: Any) -> Any:
        """
        Store object in Ray's distributed object store.

        Args:
            obj: Object to store (must be serializable)

        Returns:
            ObjectRef: Reference to the stored object
        """
        return ray.put(obj)

    def create_actor(self, klass: type, args: tuple = (), kwargs: dict | None = None, options: Any | None = None) -> Any:
        """
        Create a long-lived stateful Ray actor.

        Args:
            klass: Actor class to instantiate
            args: Positional arguments for klass.__init__
            kwargs: Keyword arguments for klass.__init__
            options: Actor creation options (ActorOptions)

        Returns:
            RayActorHandle: Wrapped Ray ActorHandle to the actor
        """
        if kwargs is None:
            kwargs = {}

        # Wrap the class as a Ray actor
        actor_cls = ray.remote(klass)

        # Apply options if provided
        if options is not None:
            actor_options = {}

            if options.name:
                actor_options["name"] = options.name

            if options.get_if_exists:
                actor_options["get_if_exists"] = options.get_if_exists

            # Translate resources from generic format to Ray-specific format
            if options.resources:
                # Extract CPU and GPU to use Ray's num_cpus/num_gpus parameters
                cpu = options.resources.get("CPU")
                gpu = options.resources.get("GPU")

                if cpu is not None:
                    actor_options["num_cpus"] = cpu
                if gpu is not None:
                    actor_options["num_gpus"] = gpu

                # Pass other custom resources through to resources dict
                other_resources = {k: v for k, v in options.resources.items() if k not in ("CPU", "GPU")}
                if other_resources:
                    actor_options["resources"] = other_resources

            # Handle actor lifetime (only set if DETACHED, EPHEMERAL is Ray's default)
            if options.lifetime == Lifetime.DETACHED:
                actor_options["lifetime"] = "detached"

            if actor_options:
                actor_cls = actor_cls.options(**actor_options)

        # Create the actor and wrap it in RayActorHandle
        raw_handle = actor_cls.remote(*args, **kwargs)
        return RayActorHandle(raw_handle)
