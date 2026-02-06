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

"""Client protocol and helpers for fray v2."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import time
from collections.abc import Generator, Sequence
from typing import Any, Protocol, runtime_checkable

from fray.v2.actor import ActorGroup, ActorHandle
from fray.v2.types import JobRequest, JobStatus, ResourceConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class JobHandle(Protocol):
    @property
    def job_id(self) -> str: ...

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        """Block until job completes."""
        ...

    def status(self) -> JobStatus: ...

    def terminate(self) -> None: ...


@runtime_checkable
class Client(Protocol):
    def submit(self, request: JobRequest) -> JobHandle:
        """Submit a job for execution. Returns immediately."""
        ...

    def create_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs: Any,
    ) -> ActorHandle:
        """Create a named actor instance. Returns a handle immediately."""
        ...

    def create_actor_group(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        count: int,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs: Any,
    ) -> ActorGroup:
        """Create N instances of an actor, returning a group handle."""
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the client and all managed resources."""
        ...


class JobFailed(RuntimeError):
    """Raised when a job fails during wait_all with raise_on_failure=True."""

    def __init__(self, job_id: str, status: JobStatus):
        self.job_id = job_id
        self.failed_status = status
        super().__init__(f"Job {job_id} finished with status {status}")


def wait_all(
    jobs: Sequence[JobHandle],
    *,
    timeout: float | None = None,
    raise_on_failure: bool = True,
) -> list[JobStatus]:
    """Wait for all jobs to complete, monitoring concurrently.

    Args:
        jobs: Job handles to wait for.
        timeout: Maximum seconds to wait. None means wait forever.
        raise_on_failure: If True, raise JobFailed on the first failed job.

    Returns:
        Final status for each job, in the same order as the input.
    """
    if not jobs:
        return []

    results: list[JobStatus | None] = [None] * len(jobs)
    remaining = set(range(len(jobs)))
    start = time.monotonic()
    sleep_secs = 0.05
    max_sleep_secs = 2.0

    while remaining:
        if timeout is not None and (time.monotonic() - start) > timeout:
            raise TimeoutError(f"wait_all timed out after {timeout}s with {len(remaining)} jobs remaining")

        for i in list(remaining):
            s = jobs[i].status()
            if JobStatus.finished(s):
                results[i] = s
                remaining.discard(i)
                if raise_on_failure and s in (JobStatus.FAILED, JobStatus.STOPPED):
                    raise JobFailed(jobs[i].job_id, s)

        if remaining:
            time.sleep(sleep_secs)
            sleep_secs = min(sleep_secs * 1.5, max_sleep_secs)

    return results  # type: ignore[return-value]


_current_client_var: contextvars.ContextVar[Client | None] = contextvars.ContextVar("_current_client_var", default=None)


def current_client() -> Client:
    """Return the current fray Client.

    Resolution order:
        1. Explicitly set client (via set_current_client)
        2. Auto-detect Iris environment (get_iris_ctx() returns context)
        3. Auto-detect Ray environment (ray.is_initialized())
        4. LocalClient() default
    """
    client = _current_client_var.get()
    if client is not None:
        return client

    try:
        from iris.client.client import get_iris_ctx

        ctx = get_iris_ctx()
        if ctx is not None:
            from fray.v2.iris_backend import FrayIrisClient

            return FrayIrisClient.from_iris_client(ctx.client)
    except ImportError:
        pass  # Iris not installed

    # Auto-detect Ray environment
    try:
        import ray

        if ray.is_initialized():
            from fray.v2.ray_backend.backend import RayClient

            return RayClient()
    except ImportError:
        pass  # Ray not installed

    from fray.v2.local_backend import LocalClient

    return LocalClient()


@contextlib.contextmanager
def set_current_client(client: Client) -> Generator[Client, None, None]:
    """Context manager that sets the current client and restores on exit."""
    token = _current_client_var.set(client)
    try:
        yield client
    finally:
        _current_client_var.reset(token)
