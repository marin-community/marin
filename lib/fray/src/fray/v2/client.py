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
import os
import time
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from fray.v2.types import JobRequest, JobStatus, ResourceConfig

if TYPE_CHECKING:
    from fray.v2.actor import ActorGroup, ActorHandle

logger = logging.getLogger(__name__)

_current_client_var: contextvars.ContextVar[Client | None] = contextvars.ContextVar("_current_client_var", default=None)


# ---------------------------------------------------------------------------
# JobHandle protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class JobHandle(Protocol):
    @property
    def job_id(self) -> str: ...

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        """Block until job completes."""
        ...

    def status(self) -> JobStatus: ...

    def terminate(self) -> None: ...


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# wait_all
# ---------------------------------------------------------------------------


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

    Polls all jobs in a loop rather than waiting sequentially, so a failure
    in a later job is detected without waiting for earlier jobs to finish.

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


# ---------------------------------------------------------------------------
# current_client / set_current_client
# ---------------------------------------------------------------------------


def _parse_client_spec(spec: str) -> Client:
    """Parse a FRAY_CLIENT_SPEC string into a Client instance.

    Supported formats:
        "local"              → LocalClient()
        "local?threads=4"    → LocalClient(max_threads=4)
        "ray"                → NotImplementedError
        "iris://host:port"   → NotImplementedError
    """
    from fray.v2.local import LocalClient

    parsed = urlparse(spec if "://" in spec else f"fray://{spec}")
    scheme = parsed.scheme if "://" in spec else spec.split("?")[0]

    if scheme == "local":
        params = parse_qs(parsed.query if "://" in spec else (spec.split("?", 1)[1] if "?" in spec else ""))
        threads = int(params["threads"][0]) if "threads" in params else 8
        return LocalClient(max_threads=threads)
    elif scheme == "ray":
        raise NotImplementedError("Ray client is not yet supported")
    elif scheme == "iris":
        raise NotImplementedError("Iris client is not yet supported")
    else:
        raise ValueError(f"Unknown FRAY_CLIENT_SPEC scheme: {scheme!r}")


def current_client() -> Client:
    """Return the current fray Client.

    Resolution order:
        1. Explicitly set client (via set_current_client)
        2. FRAY_CLIENT_SPEC environment variable
        3. LocalClient() default
    """
    client = _current_client_var.get()
    if client is not None:
        return client

    spec = os.environ.get("FRAY_CLIENT_SPEC")
    if spec is not None:
        return _parse_client_spec(spec)

    from fray.v2.local import LocalClient

    return LocalClient()


@contextlib.contextmanager
def set_current_client(client: Client) -> Generator[Client, None, None]:
    """Context manager that sets the current client and restores on exit."""
    token = _current_client_var.set(client)
    try:
        yield client
    finally:
        _current_client_var.reset(token)
