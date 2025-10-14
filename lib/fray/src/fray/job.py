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

"""Job context interface for task execution within a job."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class JobContext(ABC):
    """
    Context for running tasks within a job.

    Provides access to distributed task scheduling, object storage, and actor
    creation. All tasks within a job share the same execution environment and
    resources.

    This is the primary interface for interacting with the distributed system
    from within a job. Similar to Ray's implicit global context, but explicit
    and swappable.
    """

    @abstractmethod
    def create_task(self, fn: Callable, args: tuple = (), kwargs: dict | None = None, options: Any | None = None) -> Any:
        """
        Schedule a task to run asynchronously.

        Args:
            fn: Function to execute as a task
            args: Positional arguments to pass to fn
            kwargs: Keyword arguments to pass to fn
            options: Task creation options (TaskOptions: resources, runtime_env, name)

        Returns:
            ObjectRef: Reference to the future result

        Example:
            Basic task:
                def add(a, b):
                    return a + b

                ref = ctx.create_task(add, args=(2, 3))
                result = ctx.get(ref)  # Returns 5

            With keyword arguments:
                ref = ctx.create_task(process, args=(data,), kwargs={"verbose": True})

            With resource constraints:
                from fray import TaskOptions

                options = TaskOptions(resources={"CPU": 4, "memory": 8*1024**3})
                ref = ctx.create_task(process_large_file, args=(data,), options=options)

            With runtime environment:
                from fray import TaskOptions, RuntimeEnv

                options = TaskOptions(
                    runtime_env=RuntimeEnv(package_requirements=["scipy>=1.10"])
                )
                ref = ctx.create_task(scientific_compute, args=(data,), options=options)
        """
        pass

    @abstractmethod
    def get(self, ref: Any) -> Any:
        """
        Block and retrieve result from an ObjectRef.

        Args:
            ref: Single ObjectRef or list of ObjectRefs

        Returns:
            The computed value(s)

        Example:
            refs = [ctx.remote(fn, i) for i in range(10)]
            results = ctx.get(refs)  # Blocks until all complete
        """
        pass

    @abstractmethod
    def wait(self, refs: list[Any], num_returns: int = 1, timeout: float | None = None) -> tuple[list[Any], list[Any]]:
        """
        Wait for tasks to complete.

        Args:
            refs: List of ObjectRefs to wait on
            num_returns: Number of tasks to wait for before returning
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            Tuple of (done_refs, not_done_refs)

        Example:
            refs = [ctx.remote(slow_task) for _ in range(10)]
            done, not_done = ctx.wait(refs, num_returns=3, timeout=5.0)
            # Returns as soon as 3 tasks complete or 5 seconds pass
        """
        pass

    @abstractmethod
    def put(self, obj: Any) -> Any:
        """
        Store object in distributed object store.

        Args:
            obj: Object to store (must be serializable)

        Returns:
            ObjectRef: Reference to the stored object

        Example:
            large_data = {"key": [1, 2, 3] * 1000}
            ref = ctx.put(large_data)
            # Can pass ref to multiple tasks without copying data
        """
        pass

    @abstractmethod
    def create_actor(self, klass: type, args: tuple = (), kwargs: dict | None = None, options: Any | None = None) -> Any:
        """
        Create a long-lived stateful actor.

        Actors are stateful services that persist across multiple method calls.
        Unlike tasks, actor methods can mutate internal state.

        Args:
            klass: Actor class to instantiate
            args: Positional arguments for klass.__init__
            kwargs: Keyword arguments for klass.__init__
            options: Actor creation options (ActorOptions), including name, resources, and scheduling

        Returns:
            ActorRef: Handle to the actor
        """
        pass
