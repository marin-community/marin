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

"""Execution contexts for distributed and parallel computing.

This provides object storage and task management functions for use within a job.
"""

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

try:
    import ray
except ImportError:
    ray = None

from fray.fray_rpc import RustyContext

logger = logging.getLogger(__name__)

# Context variable for the current job context, shared across all calls to fray_job_ctx().
_job_context: ContextVar[Any | None] = ContextVar("fray_job_context", default=None)


class JobContext(Protocol):
    """Protocol for execution contexts that abstract put/get/run/wait primitives.

    This allows different backends (Ray, ThreadPool, Sync) to share the same
    execution logic while using different execution strategies.
    """

    def put(self, obj: Any) -> Any:
        """Store an object and return a reference to it."""
        ...

    def get(self, ref: Any) -> Any:
        """Retrieve an object from its reference."""
        ...

    def run(self, fn: Callable, *args) -> Any:
        """Execute a function with arguments and return a future."""
        ...

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete."""
        ...

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
        """Create an actor (stateful service) within the execution context."""
        ...


class ActorHandle:
    """Base class for actor handles."""

    def __getattr__(self, method_name: str) -> "ActorMethod":
        raise NotImplementedError


class ActorMethod:
    """Base class for actor method wrappers."""

    def remote(self, *args, **kwargs) -> Any:
        raise NotImplementedError


@dataclass
class ContextConfig:
    """Configuration for execution context creation.

    Attributes:
        context_type: Type of context (ray, threadpool, sync, or auto)
        max_workers: Maximum number of worker threads (threadpool only)
        memory: Memory requirement per task in bytes (ray only)
        num_cpus: Number of CPUs per task (ray only)
        num_gpus: Number of GPUs per task (ray only)
        ray_options: Additional Ray remote options (ray only)
    """

    context_type: Literal["ray", "threadpool", "sync", "auto"]
    max_workers: int = 1
    memory: int | None = None
    num_cpus: float | None = None
    num_gpus: float | None = None
    ray_options: dict = field(default_factory=dict)


@contextmanager
def fray_default_job_ctx(ctx: JobContext):
    """Set the default job context for the duration of the context."""
    old_ctx = _job_context.get()
    _job_context.set(ctx)
    try:
        yield ctx
    finally:
        _job_context.set(old_ctx)


def get_default_job_ctx() -> JobContext:
    """Get the current default job context, creating one if unset."""
    ctx = _job_context.get()
    if ctx is None:
        ctx = create_job_ctx(context_type="auto")
    return ctx


def create_job_ctx(
    context_type: Literal["ray", "threadpool", "sync", "rpc", "auto"] = "auto",
    max_workers: int = 1,
    coordinator_addr: str = "127.0.0.1:50051",
    **ray_options,
) -> JobContext:
    """Create a new job context.

    Args:
        context_type: Type of context (ray, threadpool, sync, rpc, or auto)
        max_workers: Maximum number of worker threads (threadpool only)
        coordinator_addr: Coordinator address for rpc backend (default: "127.0.0.1:50051")
        **ray_options: Additional Ray remote options
    """
    if context_type == "auto":
        if ray and ray.is_initialized():
            context_type = "ray"
        else:
            context_type = "threadpool"

    if context_type == "sync":
        from fray.job.sync_ctx import SyncContext

        return SyncContext()
    elif context_type == "threadpool":
        from fray.job.thread_ctx import ThreadContext

        workers = min(max_workers, os.cpu_count() or 1)
        return ThreadContext(max_workers=workers)
    elif context_type == "ray":
        from fray.job.ray_ctx import RayContext

        return RayContext(ray_options=ray_options)
    elif context_type == "rpc":
        return RustyContext(coordinator_addr)
    else:
        raise ValueError(f"Unknown context type: {context_type}. Supported: 'ray', 'threadpool', 'sync', 'rpc'")


class SimpleActor:
    """Test actor for basic actor functionality. (Ray cannot import from test modules)."""

    def __init__(self, value: int):
        self.value = value
        self.call_count = 0

    def increment(self, amount: int = 1) -> int:
        self.call_count += 1
        self.value += amount
        return self.value

    def get_value(self) -> int:
        return self.value
