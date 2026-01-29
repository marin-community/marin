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

"""Cluster protocol and factory functions for Fray v2.

This module provides:
- Job: Protocol for job handles
- Cluster: Protocol for cluster operations
- Resolver: Protocol for actor discovery
- ActorPool: Protocol for actor RPC
- WorkerPool: Protocol for task dispatch
- current_cluster(): Factory for getting cluster from environment
- create_cluster(): Factory for creating cluster from spec string
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from fray.v2.types import (
    Entrypoint,
    EnvironmentSpec,
    JobId,
    JobStatus,
    Namespace,
    ResourceSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class Job(Protocol):
    """Job handle returned by Cluster.submit().

    Provides convenient methods for job operations like waiting,
    checking status, and termination.
    """

    @property
    def job_id(self) -> JobId:
        """Unique job identifier."""
        ...

    def status(self) -> JobStatus:
        """Get current job status."""
        ...

    def wait(
        self,
        timeout: float = 300.0,
        *,
        stream_logs: bool = False,
        raise_on_failure: bool = True,
    ) -> JobStatus:
        """Wait for job to complete.

        Args:
            timeout: Maximum wait time in seconds
            stream_logs: If True, stream logs while waiting
            raise_on_failure: If True, raise exception on non-SUCCESS terminal state

        Returns:
            Final job status

        Raises:
            TimeoutError: Job didn't complete in time
            RuntimeError: Job failed and raise_on_failure=True
        """
        ...

    def terminate(self) -> None:
        """Terminate this job."""
        ...


@runtime_checkable
class BroadcastResult(Protocol):
    """Result of broadcasting a method call to all actors."""

    def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Wait for all results, returning list (may include exceptions)."""
        ...

    def wait_any(self, timeout: float | None = None) -> Any:
        """Wait for first result."""
        ...


@runtime_checkable
class ActorPool(Protocol):
    """Pool of actors for RPC calls.

    Provides load-balanced calls via call() and broadcast via broadcast().
    """

    @property
    def size(self) -> int:
        """Current number of actors in the pool."""
        ...

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None:
        """Wait until pool has at least min_size actors.

        Args:
            min_size: Minimum number of actors required
            timeout: Maximum time to wait

        Raises:
            TimeoutError: If timeout expires before min_size reached
        """
        ...

    def call(self) -> Any:
        """Get proxy for single-actor round-robin calls.

        Returns a proxy that routes method calls to one actor in the pool.
        """
        ...

    def broadcast(self) -> Any:
        """Get proxy for broadcasting to all actors.

        Returns a proxy that calls all actors and returns BroadcastResult.
        """
        ...


@runtime_checkable
class Resolver(Protocol):
    """Actor discovery protocol.

    Maps actor names to ActorPool instances.
    """

    def lookup(self, name: str) -> ActorPool:
        """Look up actors by name and return a pool.

        Always returns a pool, even if empty. Use pool.wait_for_size()
        to block until actors are available.

        Args:
            name: Actor name to look up

        Returns:
            ActorPool for the named actor(s)
        """
        ...


@runtime_checkable
class WorkerPool(Protocol):
    """Pool of stateless workers for task dispatch.

    Provides submit() for task execution and supports context manager protocol.
    """

    @property
    def size(self) -> int:
        """Number of workers currently available."""
        ...

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a task for execution.

        Args:
            fn: Callable to execute (must be picklable)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future that resolves to the function's return value
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        ...

    def __enter__(self) -> WorkerPool:
        """Enter context manager."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Exit context manager, shutting down the pool."""
        ...


@runtime_checkable
class Cluster(Protocol):
    """Cluster protocol for job lifecycle management.

    This is the main interface for interacting with a compute cluster.
    Supports job submission, actor discovery, and task dispatch.
    """

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> Job:
        """Submit a job to the cluster.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Optional environment configuration

        Returns:
            Job handle for the submitted job

        Raises:
            ValueError: If name contains '/'
        """
        ...

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Args:
            job_id: Job identifier to terminate
        """
        ...

    def list_jobs(self) -> list[Job]:
        """List all jobs managed by this cluster.

        Returns:
            List of Job handles
        """
        ...

    def resolver(self) -> Resolver:
        """Get resolver for actor discovery.

        Returns:
            Resolver for looking up actors by name
        """
        ...

    def worker_pool(
        self,
        num_workers: int,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> WorkerPool:
        """Create a worker pool for task dispatch.

        Args:
            num_workers: Number of worker jobs to launch
            resources: Resource requirements per worker
            environment: Optional environment configuration

        Returns:
            WorkerPool context manager
        """
        ...

    @property
    def namespace(self) -> Namespace:
        """The namespace for this cluster connection."""
        ...


# Context variable for current cluster
_cluster_context: ContextVar[Cluster | None] = ContextVar("fray_v2_cluster", default=None)


def create_cluster(spec: str = "local") -> Cluster:
    """Create a cluster from a spec string.

    Spec formats:
    - "local" -> LocalCluster (in-process, thread-based)
    - "iris" or "iris?address=http://..." -> IrisCluster
    - "ray" or "ray?namespace=xyz" -> RayCluster

    Args:
        spec: Cluster specification string

    Returns:
        Cluster instance

    Raises:
        ValueError: If spec format is invalid or backend unavailable
    """
    parsed = urlparse(spec if "://" in spec or "?" in spec else f"{spec}://")
    backend = parsed.scheme or parsed.path or "local"
    params = parse_qs(parsed.query)

    # Flatten single-value params
    options = {k: v[0] if len(v) == 1 else v for k, v in params.items()}

    if backend == "local":
        from fray.v2.backends.local import LocalCluster

        return LocalCluster(**options)  # type: ignore[return-value]

    elif backend == "iris":
        from fray.v2.backends.iris import IrisCluster

        address = options.pop("address", None)
        workspace = options.pop("workspace", None)
        if address:
            return IrisCluster.remote(address, workspace=workspace, **options)  # type: ignore[return-value]
        else:
            return IrisCluster.local(**options)  # type: ignore[return-value]

    elif backend == "ray":
        from fray.v2.backends.ray import RayCluster

        return RayCluster(**options)  # type: ignore[return-value]

    else:
        raise ValueError(f"Unknown cluster backend: {backend}. Supported: local, iris, ray")


def current_cluster() -> Cluster:
    """Get the current cluster from environment or context.

    Reads FRAY_CLUSTER_SPEC environment variable:
    - "local" or unset: LocalCluster
    - "iris" or "iris?address=http://...": IrisCluster
    - "ray" or "ray?namespace=xyz": RayCluster

    Returns:
        Current Cluster instance

    Raises:
        ValueError: If FRAY_CLUSTER_SPEC is invalid
    """
    # Check context first
    cluster = _cluster_context.get()
    if cluster is not None:
        return cluster

    # Fall back to environment variable
    spec = os.environ.get("FRAY_CLUSTER_SPEC", "local")
    cluster = create_cluster(spec)
    _cluster_context.set(cluster)
    return cluster


def set_current_cluster(cluster: Cluster | None) -> None:
    """Set the current cluster in context.

    Args:
        cluster: Cluster to set as current, or None to clear
    """
    _cluster_context.set(cluster)
