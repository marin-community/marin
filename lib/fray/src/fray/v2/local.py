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
from typing import Any

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
    ) -> Any:
        raise NotImplementedError("LocalClient.create_actor is implemented in Phase 1b")

    def create_actor_group(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        count: int,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("LocalClient.create_actor_group is implemented in Phase 1b")

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
