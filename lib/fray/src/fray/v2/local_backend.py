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

"""LocalClient: in-process fray v2 backend for development and testing."""

from __future__ import annotations

import logging
import subprocess
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast

from fray.v2.actor import ActorFuture, ActorGroup, ActorHandle
from fray.v2.types import (
    BinaryEntrypoint,
    CallableEntrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
)

logger = logging.getLogger(__name__)


class LocalJobHandle:
    """Job handle backed by a concurrent.futures.Future."""

    def __init__(self, job_id: str, future: Future[None]):
        self._job_id = job_id
        self._future = future
        self._terminated = threading.Event()

    @property
    def job_id(self) -> str:
        return self._job_id

    def status(self) -> JobStatus:
        if self._terminated.is_set():
            return JobStatus.STOPPED
        if not self._future.done():
            return JobStatus.RUNNING
        exc = self._future.exception()
        if exc is not None:
            return JobStatus.FAILED
        return JobStatus.SUCCEEDED

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        """Block until the job completes or timeout expires."""
        try:
            self._future.result(timeout=timeout)
        except Exception:
            if raise_on_failure:
                raise
        return self.status()

    def terminate(self) -> None:
        self._terminated.set()
        self._future.cancel()


def _run_callable(entry: CallableEntrypoint) -> None:
    entry.callable(*entry.args, **entry.kwargs)


def _run_binary(entry: BinaryEntrypoint) -> None:
    subprocess.run(
        [entry.command, *entry.args],
        check=True,
        capture_output=True,
        text=True,
    )


class LocalClient:
    """In-process Client implementation for development and testing.

    Runs CallableEntrypoints in threads and BinaryEntrypoints as subprocesses.
    """

    def __init__(self, max_threads: int = 8):
        self._executor = ThreadPoolExecutor(max_workers=max_threads)
        self._jobs: list[LocalJobHandle] = []

    def submit(self, request: JobRequest) -> LocalJobHandle:
        entry = request.entrypoint
        if entry.callable_entrypoint is not None:
            future = self._executor.submit(_run_callable, entry.callable_entrypoint)
        elif entry.binary_entrypoint is not None:
            future = self._executor.submit(_run_binary, entry.binary_entrypoint)
        else:
            raise ValueError("JobRequest entrypoint must have either callable_entrypoint or binary_entrypoint")

        job_id = f"local-{request.name}-{uuid.uuid4().hex[:8]}"
        handle = LocalJobHandle(job_id, future)
        self._jobs.append(handle)
        return handle

    def create_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs: Any,
    ) -> LocalActorHandle:
        """Create an in-process actor, returning a handle immediately."""
        group = self.create_actor_group(actor_class, *args, name=name, count=1, resources=resources, **kwargs)
        return group.wait_ready()[0]  # type: ignore[return-value]

    def create_actor_group(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        count: int,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs: Any,
    ) -> ActorGroup:
        """Create N in-process actor instances, returning a group handle."""
        handles: list[LocalActorHandle] = []
        jobs: list[LocalJobHandle] = []
        for i in range(count):
            instance = actor_class(*args, **kwargs)
            handle = LocalActorHandle(instance)
            handles.append(handle)
            # Create a synthetic job handle that is immediately succeeded
            future: Future[None] = Future()
            future.set_result(None)
            job_id = f"local-actor-{name}-{i}-{uuid.uuid4().hex[:8]}"
            jobs.append(LocalJobHandle(job_id, future))
        return LocalActorGroup(cast(list[ActorHandle], handles), jobs)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


class LocalActorHandle:
    """In-process actor handle.

    Actors are responsible for their own thread safety. This matches Iris/Ray
    behavior where actor methods can be called concurrently and the actor
    implementation must handle synchronization internally.
    """

    def __init__(self, instance: Any):
        self._instance = instance
        self._executor = ThreadPoolExecutor(max_workers=16)

    def __getattr__(self, method_name: str) -> LocalActorMethod:
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        method = getattr(self._instance, method_name)
        if not callable(method):
            raise AttributeError(f"{method_name} is not callable on {type(self._instance).__name__}")
        return LocalActorMethod(method, self._executor)


class LocalActorMethod:
    """Wraps a method on a local actor."""

    def __init__(self, method: Any, executor: ThreadPoolExecutor):
        self._method = method
        self._executor = executor

    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        """Submit method call to thread pool, returning a future."""
        return self._executor.submit(self._method, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call method synchronously."""
        return self._method(*args, **kwargs)


class LocalActorGroup:
    """ActorGroup for local in-process actors. All actors are ready immediately."""

    def __init__(self, handles: list[ActorHandle], jobs: list[LocalJobHandle]):
        self._handles = handles
        self._jobs = jobs
        self._yielded = False

    @property
    def ready_count(self) -> int:
        """All local actors are ready immediately after creation."""
        return len(self._handles)

    def wait_ready(self, count: int | None = None, timeout: float = 300.0) -> list[ActorHandle]:
        """Return ready actor handles. Local actors are ready immediately."""
        if count is None:
            count = len(self._handles)
        self._yielded = True
        return self._handles[:count]

    def discover_new(self) -> list[ActorHandle]:
        """Return handles not yet yielded. After wait_ready, returns empty."""
        if self._yielded:
            return []
        self._yielded = True
        return self._handles

    def shutdown(self) -> None:
        """Terminate all local actors."""
        for job in self._jobs:
            job.terminate()
        for handle in self._handles:
            handle._executor.shutdown(wait=False)
