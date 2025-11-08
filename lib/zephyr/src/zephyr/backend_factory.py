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

"""Factory for creating backend instances from configuration."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import replace
from typing import Literal

import humanfriendly

from zephyr.backends import Backend, BackendConfig, RayBackend, SyncBackend, ThreadPoolBackend

logger = logging.getLogger(__name__)

_backend_context: ContextVar[Backend | None] = ContextVar("zephyr_backend", default=None)


def create_backend(
    backend_type: Literal["ray", "threadpool", "sync"],
    max_parallelism: int = 100,
    memory: str | None = None,
    num_cpus: float | None = None,
    num_gpus: float | None = None,
    chunk_size: int = 1000,
    dry_run: bool = False,
    **ray_options,
) -> Backend:
    """Create backend instance from configuration parameters.

    Args:
        backend_type: Type of backend (ray, threadpool, or sync)
        max_parallelism: Maximum number of concurrent tasks (default: 100)
        memory: Memory requirement per task (e.g., "2GB", "512MB")
        num_cpus: Number of CPUs per task for Ray backend
        num_gpus: Number of GPUs per task for Ray backend
        chunk_size: Items per chunk within a shard (default: 1000)
        dry_run: If True, show optimization plan without executing
        **ray_options: Additional Ray remote options (e.g., max_retries=3)

    Returns:
        Backend instance

    Raises:
        ValueError: If backend_type is invalid

    Examples:
        >>> backend = create_backend("sync")
        >>> backend = create_backend("ray", max_parallelism=100, memory="2GB")
        >>> backend = create_backend("ray", max_parallelism=10, max_retries=3)
    """
    # Parse memory string to bytes
    memory_bytes = humanfriendly.parse_size(memory, binary=True) if memory else None

    config = BackendConfig(
        backend_type=backend_type,
        max_parallelism=max_parallelism,
        memory=memory_bytes,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        ray_options=ray_options,
        dry_run=dry_run,
        chunk_size=chunk_size,
    )

    if config.backend_type == "ray":
        return RayBackend(config)
    elif config.backend_type == "threadpool":
        return ThreadPoolBackend(config)
    elif config.backend_type == "sync":
        return SyncBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {config.backend_type}. Supported: 'ray', 'threadpool', 'sync'")


def set_flow_backend(backend: Backend) -> None:
    """Set the current backend for this context.

    Used by the zephyr launcher to inject backends into user scripts.

    Args:
        backend: Backend instance to use for dataset execution
    """
    _backend_context.set(backend)


def flow_backend(
    max_parallelism: int | None = None,
    memory: str | None = None,
    num_cpus: float | None = None,
    num_gpus: float | None = None,
    chunk_size: int | None = None,
    dry_run: bool | None = None,
    **backend_options,
) -> Backend:
    """Get the current backend from context, or create a new one with custom parameters.

    If no parameters are provided, returns the current backend from context (or a default
    ThreadPoolBackend if none is configured).

    If parameters are provided, creates a new backend of the same type as the current
    backend with the specified parameters.

    Args:
        max_parallelism: Maximum number of concurrent tasks
        memory: Memory requirement per task (e.g., "2GB", "512MB")
        num_cpus: Number of CPUs per task for Ray backend
        num_gpus: Number of GPUs per task for Ray backend
        chunk_size: Items per chunk within a shard
        dry_run: If True, show optimization plan without executing
        **backend_options: Additional backend options (e.g., max_retries=3 for Ray)

    Returns:
        Backend instance

    Examples:
        >>> from zephyr import flow_backend, Dataset
        >>> # Get current backend
        >>> backend = flow_backend()
        >>> pipeline = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
        >>> list(backend.execute(pipeline))

        >>> # Create new backend with custom parameters
        >>> backend = flow_backend(max_parallelism=1000, memory="2GB")
    """
    current = _backend_context.get()

    # No parameters provided: return current backend or default
    has_params = (
        any(v is not None for v in [max_parallelism, memory, num_cpus, num_gpus, chunk_size, dry_run]) or backend_options
    )
    if not has_params:
        if current is None:
            logger.warning("No backend configured in context, using ThreadPoolBackend as default.")
            return create_backend("threadpool")
        return current

    # Parameters provided: create new backend with merged config
    if current is None:
        # No current backend, create default threadpool
        current = create_backend("threadpool")

    # Build override dict, only including non-None values
    overrides = {}
    if max_parallelism is not None:
        overrides["max_parallelism"] = max_parallelism
    if memory is not None:
        overrides["memory"] = humanfriendly.parse_size(memory, binary=True)
    if num_cpus is not None:
        overrides["num_cpus"] = num_cpus
    if num_gpus is not None:
        overrides["num_gpus"] = num_gpus
    if chunk_size is not None:
        overrides["chunk_size"] = chunk_size
    if dry_run is not None:
        overrides["dry_run"] = dry_run
    if backend_options:
        # Merge ray_options if present
        if "ray_options" in overrides or current.config.ray_options:
            merged_ray_options = {**current.config.ray_options, **backend_options}
            overrides["ray_options"] = merged_ray_options
        else:
            overrides["ray_options"] = backend_options

    # Clone backend with merged config
    new_config = replace(current.config, **overrides)
    return type(current)(new_config)  # type: ignore
