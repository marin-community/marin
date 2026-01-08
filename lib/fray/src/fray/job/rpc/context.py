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

"""RPC-based execution context for distributed computing.

This module implements FrayContext, a JobContext that communicates with a FrayController
via Connect RPC to execute tasks across distributed workers.

Usage
-----
Create a FrayContext by specifying the controller address::

    from fray.job.rpc.context import FrayContext

    ctx = FrayContext("http://localhost:50051")

Or use the factory function::

    from fray.job.context import create_job_ctx

    ctx = create_job_ctx("fray", controller_address="http://localhost:50051")

Submit tasks for execution::

    def process_data(data):
        return data * 2

    future = ctx.run(process_data, [1, 2, 3])
    result = ctx.get(future)  # Blocks until task completes

Wait for multiple tasks::

    futures = [ctx.run(process_data, i) for i in range(10)]
    ready, pending = ctx.wait(futures, num_returns=5)

Implementation Details
----------------------
FrayContext follows the storage-first execution model - objects are passed by value
rather than being stored in a distributed object store. The put() method returns the
object unchanged, and get() unwraps _FrayFuture objects by polling the controller
for task completion.

Tasks are serialized using cloudpickle and submitted to the controller via Connect RPC.
The controller queues tasks and assigns them to registered workers. Workers execute
tasks and report results back to the controller, which clients poll to retrieve results.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import cloudpickle
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayControllerClientSync

__all__ = ["FrayContext", "_FrayActorHandle", "_FrayActorMethod", "_FrayFuture"]


class _FrayFuture:
    """Future wrapper for RPC-based task execution."""

    def __init__(self, task_id: str, client: FrayControllerClientSync):
        self._task_id = task_id
        self._client = client
        self._result = None
        self._done = False
        self._error: str | None = None

    def result(self, timeout: float | None = None) -> Any:
        """Get task result, polling until completion."""
        if not self._done:
            handle = fray_pb2.TaskHandle(task_id=self._task_id)  # type: ignore
            start_time = time.time()
            sleep_time = 0.001  # Start at 1ms

            while True:
                status_handle = self._client.get_task_status(handle)

                if status_handle.status == fray_pb2.TASK_STATUS_COMPLETED:  # type: ignore
                    break
                elif status_handle.status == fray_pb2.TASK_STATUS_FAILED:  # type: ignore
                    # Fetch the full task result to get serialized exception
                    task_result = self._client.get_task_result(handle)

                    # If we have a serialized error, deserialize and re-raise the original exception
                    if task_result.serialized_error:
                        original_exception = cloudpickle.loads(task_result.serialized_error)
                        raise original_exception

                    # Fallback to RuntimeError if no serialized error (backward compatibility)
                    self._error = status_handle.error
                    raise RuntimeError(f"Task {self._task_id} failed: {self._error}")

                if timeout is not None and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Task {self._task_id} timed out after {timeout}s")

                time.sleep(sleep_time)
                # Exponential backoff: multiply by 1.5, cap at 100ms
                sleep_time = min(sleep_time * 1.5, 0.1)

            # Fetch result
            task_result = self._client.get_task_result(handle)
            self._result = cloudpickle.loads(task_result.serialized_result)
            self._done = True

        return self._result

    def done(self) -> bool:
        """Check if task is complete."""
        if self._done:
            return True

        handle = fray_pb2.TaskHandle(task_id=self._task_id)  # type: ignore
        status_handle = self._client.get_task_status(handle)

        # Don't set _done here - only result() should set it after fetching
        return status_handle.status in (fray_pb2.TASK_STATUS_COMPLETED, fray_pb2.TASK_STATUS_FAILED)  # type: ignore


class _FrayActorMethod:
    """Actor method wrapper that provides remote() call interface."""

    def __init__(
        self,
        actor_id: str,
        method_name: str,
        client: FrayControllerClientSync,
        max_retries: int = 10,
        base_delay: float = 0.01,
        max_delay: float = 1.0,
    ):
        self._actor_id = actor_id
        self._method_name = method_name
        self._client = client
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    def remote(self, *args, **kwargs) -> _FrayFuture:
        """Call actor method remotely with retry on UNAVAILABLE, returns future for async execution.

        Implements exponential backoff retry for UNAVAILABLE errors (e.g., during actor restart).
        Starting delay is 10ms, doubling each retry, capped at 1s, with default max 10 retries.
        """
        payload = {"method": self._method_name, "args": args, "kwargs": kwargs}
        serialized_call = cloudpickle.dumps(payload)

        call = fray_pb2.ActorCall(  # type: ignore
            actor_id=self._actor_id,
            serialized_call=serialized_call,
        )

        for attempt in range(self._max_retries):
            try:
                task_handle = self._client.call_actor(call)
                return _FrayFuture(task_handle.task_id, self._client)
            except ConnectError as e:
                if e.code == Code.UNAVAILABLE and attempt < self._max_retries - 1:
                    # Exponential backoff: start at base_delay, double each time, cap at max_delay
                    delay = min(self._base_delay * (2**attempt), self._max_delay)
                    time.sleep(delay)
                else:
                    # Re-raise if not UNAVAILABLE or max retries exceeded
                    raise

        # Should never reach here, but satisfy type checker
        raise RuntimeError("Unexpected state in retry loop")


class _FrayActorHandle:
    """Actor handle for remote method calls.

    Provides __getattr__ to dynamically return method wrappers for any method name.
    """

    def __init__(
        self,
        actor_id: str,
        client: FrayControllerClientSync,
        max_retries: int = 10,
        base_delay: float = 0.01,
        max_delay: float = 1.0,
    ):
        self._actor_id = actor_id
        self._client = client
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    def __getattr__(self, method_name: str) -> _FrayActorMethod:
        """Return method wrapper for any requested method."""
        return _FrayActorMethod(
            self._actor_id,
            method_name,
            self._client,
            self._max_retries,
            self._base_delay,
            self._max_delay,
        )


class FrayContext:
    """Execution context for RPC-based distributed computing."""

    def __init__(self, controller_address: str):
        if not controller_address.startswith("http"):
            controller_address = f"http://{controller_address}"
        self._client = FrayControllerClientSync(controller_address)

    def put(self, obj: Any) -> Any:
        """Storage-first model - return object unchanged."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, unwrapping _FrayFuture if needed."""
        if isinstance(ref, _FrayFuture):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> _FrayFuture:
        """Submit function for remote execution."""
        payload = {"fn": fn, "args": args}
        serialized_fn = cloudpickle.dumps(payload)

        task_spec = fray_pb2.TaskSpec(serialized_fn=serialized_fn)  # type: ignore
        handle = self._client.submit_task(task_spec)

        return _FrayFuture(handle.task_id, self._client)

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete."""
        sleep_time = 0.001  # Start at 1ms

        while True:
            ready = []
            pending = []

            for i, future in enumerate(futures):
                if isinstance(future, _FrayFuture) and future.done():
                    ready.append(future)
                    if len(ready) >= num_returns:
                        # Add remaining unprocessed futures to pending
                        pending.extend(futures[i + 1 :])
                        return ready, pending
                else:
                    pending.append(future)

            if len(ready) >= num_returns:
                return ready, pending

            time.sleep(sleep_time)
            # Exponential backoff: multiply by 1.5, cap at 100ms
            sleep_time = min(sleep_time * 1.5, 0.1)

    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["non_detached", "detached"] = "non_detached",
        preemptible: bool = True,
        **kwargs,
    ) -> _FrayActorHandle:
        """Create actor and return handle for remote method calls.

        Args:
            actor_class: The actor class to instantiate
            *args: Positional arguments for actor constructor
            name: Optional name for the actor (enables get_if_exists)
            get_if_exists: If True, return existing actor with same name
            lifetime: Actor lifetime (only "non_detached" supported in Phase 1)
            preemptible: Whether actor can be preempted (not used in Phase 1)
            **kwargs: Keyword arguments for actor constructor

        Returns:
            Actor handle for calling methods via .method.remote()
        """
        payload = {"cls": actor_class, "args": args, "kwargs": kwargs}
        serialized_actor = cloudpickle.dumps(payload)

        spec = fray_pb2.ActorSpec(  # type: ignore
            serialized_actor=serialized_actor,
            name=name or "",
            get_if_exists=get_if_exists,
        )
        handle = self._client.create_actor(spec)

        return _FrayActorHandle(handle.actor_id, self._client)
