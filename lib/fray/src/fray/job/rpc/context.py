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

from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayControllerClientSync

__all__ = ["FrayContext", "_FrayFuture"]


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
    ) -> Any:
        """Create an actor (not yet supported)."""
        raise NotImplementedError("Actors not yet supported in FrayContext")
