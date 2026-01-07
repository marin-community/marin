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
    ray = None  # type: ignore

from fray.job.ray_ctx import RayContext
from fray.job.rpc.context import FrayContext
from fray.job.sync_ctx import SyncContext
from fray.job.threadpool_ctx import ThreadContext

__all__ = [
    "ContextConfig",
    "FrayContext",
    "JobContext",
    "RayContext",
    "SimpleActor",
    "SyncContext",
    "ThreadContext",
    "_job_context",
    "create_job_ctx",
    "fray_default_job_ctx",
    "get_default_job_ctx",
]

logger = logging.getLogger(__name__)

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


@dataclass
class ContextConfig:
    """Configuration for execution context creation."""

    context_type: Literal["ray", "threadpool", "sync", "fray", "auto"]
    max_workers: int = 1
    controller_address: str | None = None
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
    context_type: Literal["ray", "threadpool", "sync", "fray", "auto"] = "auto",
    max_workers: int = 1,
    controller_address: str | None = None,
    **ray_options,
) -> JobContext:
    """Create a new job context.

    Args:
        context_type: Type of context to create. Options:
            - "ray": Use Ray for distributed execution
            - "threadpool": Use ThreadPoolExecutor for parallel execution
            - "sync": Synchronous execution (no parallelism)
            - "fray": Use Fray RPC for distributed execution
            - "auto": Automatically select Ray if initialized, else threadpool
        max_workers: Number of workers for threadpool context
        controller_address: Controller address for fray context (required if context_type="fray")
        **ray_options: Additional options for Ray context

    Returns:
        A JobContext implementation matching the requested type

    Raises:
        ValueError: If context_type is unknown or if fray context is requested without controller_address
    """
    if context_type == "auto":
        if ray and ray.is_initialized():
            context_type = "ray"
        else:
            context_type = "threadpool"

    if context_type == "sync":
        return SyncContext()
    elif context_type == "threadpool":
        workers = min(max_workers, os.cpu_count() or 1)
        return ThreadContext(max_workers=workers)
    elif context_type == "ray":
        return RayContext(ray_options=ray_options)
    elif context_type == "fray":
        if controller_address is None:
            raise ValueError("controller_address required for fray context")
        return FrayContext(controller_address)
    else:
        raise ValueError(f"Unknown context type: {context_type}. Supported: 'ray', 'threadpool', 'sync', 'fray'")


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
