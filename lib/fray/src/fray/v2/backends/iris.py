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

"""Iris backend for Fray v2.

Provides Iris-based implementations that wrap the Iris client:
- IrisCluster: Wraps IrisClient (local or remote mode)
- IrisJob: Wraps iris.client.Job
- IrisResolver: Wraps NamespacedResolver
- IrisActorPool: Wraps Iris actor pool
- IrisWorkerPool: Wraps iris.client.WorkerPool
- IrisActorServer: Re-exports iris.actor.ActorServer

This backend provides the canonical Iris semantics that Fray v2 is designed around.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from iris.actor import ActorServer as IrisActorServerImpl
from iris.actor.client import ActorClient
from iris.actor.resolver import Resolver as IrisResolverProtocol
from iris.client import IrisClient, Job as IrisJobImpl, LocalClientConfig
from iris.client.worker_pool import WorkerPool as IrisWorkerPoolImpl, WorkerPoolConfig, WorkerFuture
from iris.cluster.types import (
    Entrypoint as IrisEntrypoint,
    EnvironmentSpec as IrisEnvironmentSpec,
    JobId as IrisJobId,
    Namespace as IrisNamespace,
    ResourceSpec as IrisResourceSpec,
    tpu_device,
)
from iris.rpc import cluster_pb2

from fray.v2.types import (
    Entrypoint,
    EnvironmentSpec,
    JobId,
    JobStatus,
    Namespace,
    ResourceSpec,
)

logger = logging.getLogger(__name__)


def _v2_to_iris_entrypoint(v2: Entrypoint) -> IrisEntrypoint:
    """Convert Fray v2 Entrypoint to Iris Entrypoint."""
    return IrisEntrypoint(
        callable=v2.callable,
        args=v2.args,
        kwargs=v2.kwargs,
    )


def _v2_to_iris_resources(v2: ResourceSpec) -> IrisResourceSpec:
    """Convert Fray v2 ResourceSpec to Iris ResourceSpec."""
    # Handle device configuration
    device = None
    if v2.device_type == "tpu" and v2.device_variant:
        device = tpu_device(v2.device_variant, v2.device_count if v2.device_count > 0 else None)
    elif v2.device_type == "gpu" and v2.device_variant:
        device = cluster_pb2.DeviceConfig(
            gpu=cluster_pb2.GpuDevice(
                variant=v2.device_variant,
                count=v2.device_count if v2.device_count > 0 else 1,
            )
        )

    return IrisResourceSpec(
        cpu=v2.cpu,
        memory=v2.memory,
        disk=v2.disk,
        device=device,
        replicas=v2.replicas,
        preemptible=v2.preemptible,
        regions=v2.regions,
    )


def _v2_to_iris_environment(v2: EnvironmentSpec | None) -> IrisEnvironmentSpec | None:
    """Convert Fray v2 EnvironmentSpec to Iris EnvironmentSpec."""
    if v2 is None:
        return None

    return IrisEnvironmentSpec(
        workspace=v2.workspace,
        pip_packages=v2.pip_packages,
        env_vars=v2.env_vars,
        extras=v2.extras,
    )


def _iris_to_v2_status(iris_state: int) -> JobStatus:
    """Convert Iris job state to Fray v2 JobStatus."""
    mapping = {
        cluster_pb2.JOB_STATE_PENDING: JobStatus.PENDING,
        cluster_pb2.JOB_STATE_RUNNING: JobStatus.RUNNING,
        cluster_pb2.JOB_STATE_SUCCEEDED: JobStatus.SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED: JobStatus.FAILED,
        cluster_pb2.JOB_STATE_KILLED: JobStatus.KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED: JobStatus.FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE: JobStatus.FAILED,
    }
    return mapping.get(iris_state, JobStatus.FAILED)


class IrisJob:
    """Job handle wrapping an Iris Job."""

    def __init__(self, iris_job: IrisJobImpl):
        self._iris_job = iris_job

    @property
    def job_id(self) -> JobId:
        return JobId(str(self._iris_job.job_id))

    def status(self) -> JobStatus:
        """Get current job status."""
        iris_status = self._iris_job.status()
        return _iris_to_v2_status(iris_status.state)

    def wait(
        self,
        timeout: float = 300.0,
        *,
        stream_logs: bool = False,
        raise_on_failure: bool = True,
    ) -> JobStatus:
        """Wait for job to complete."""
        try:
            self._iris_job.wait(
                timeout=timeout,
                raise_on_failure=raise_on_failure,
                stream_logs=stream_logs,
            )
            return self.status()
        except Exception as e:
            if raise_on_failure:
                raise RuntimeError(f"Job {self.job_id} failed: {e}") from e
            return self.status()

    def terminate(self) -> None:
        """Terminate the job."""
        self._iris_job.terminate()


class IrisActorPool:
    """Actor pool wrapping Iris actor discovery."""

    def __init__(self, name: str, resolver: IrisResolverProtocol, timeout: float = 30.0):
        self._name = name
        self._resolver = resolver
        self._timeout = timeout
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        result = self._resolver.resolve(self._name)
        return len(result.endpoints)

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None:
        """Wait until pool has at least min_size actors."""
        import time

        start = time.monotonic()
        while self.size < min_size:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Pool did not reach size {min_size} in {timeout}s (current: {self.size})")
            time.sleep(0.1)

    def call(self) -> Any:
        """Get proxy for round-robin calls."""
        return ActorClient(
            resolver=self._resolver,
            name=self._name,
            timeout=self._timeout,
        )

    def broadcast(self) -> IrisBroadcastProxy:
        """Get proxy for broadcasting to all actors."""
        return IrisBroadcastProxy(self._name, self._resolver, self._timeout)


class IrisBroadcastResult:
    """Result of broadcasting to Iris actors."""

    def __init__(self, results: list[Any], errors: list[Exception | None]):
        self._results = results
        self._errors = errors

    def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Return all results."""
        return [e if e is not None else r for r, e in zip(self._results, self._errors, strict=True)]

    def wait_any(self, timeout: float | None = None) -> Any:
        """Return first successful result."""
        for result, error in zip(self._results, self._errors, strict=True):
            if error is None:
                return result
        if self._errors:
            raise self._errors[0]  # type: ignore[misc]
        raise RuntimeError("No results")


class IrisBroadcastProxy:
    """Proxy for broadcasting to all Iris actors."""

    def __init__(self, name: str, resolver: IrisResolverProtocol, timeout: float):
        self._name = name
        self._resolver = resolver
        self._timeout = timeout

    def __getattr__(self, method_name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> IrisBroadcastResult:
            result = self._resolver.resolve(self._name)
            results = []
            errors: list[Exception | None] = []

            for _endpoint in result.endpoints:
                try:
                    client = ActorClient(
                        resolver=self._resolver,
                        name=self._name,
                        timeout=self._timeout,
                    )
                    actor_method = getattr(client, method_name)
                    res = actor_method(*args, **kwargs)
                    results.append(res)
                    errors.append(None)
                except Exception as e:
                    results.append(None)
                    errors.append(e)

            return IrisBroadcastResult(results, errors)

        return method


class IrisResolver:
    """Resolver wrapping Iris NamespacedResolver."""

    def __init__(self, iris_resolver: IrisResolverProtocol, timeout: float = 30.0):
        self._iris_resolver = iris_resolver
        self._timeout = timeout

    def lookup(self, name: str) -> IrisActorPool:
        """Look up actors by name."""
        return IrisActorPool(name, self._iris_resolver, self._timeout)


class _FutureLikeFuture:
    """Wraps WorkerFuture to match concurrent.futures.Future interface."""

    def __init__(self, worker_future: WorkerFuture):
        self._worker_future = worker_future

    def result(self, timeout: float | None = None) -> Any:
        return self._worker_future.result(timeout=timeout)

    def done(self) -> bool:
        return self._worker_future.done()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._worker_future.exception()


class IrisWorkerPool:
    """Worker pool wrapping Iris WorkerPool."""

    def __init__(
        self,
        client: IrisClient,
        num_workers: int,
        resources: IrisResourceSpec,
        environment: IrisEnvironmentSpec | None = None,
    ):
        self._client = client
        self._num_workers = num_workers
        self._config = WorkerPoolConfig(
            num_workers=num_workers,
            resources=resources,
            environment=environment,
        )
        self._pool: IrisWorkerPoolImpl | None = None

    @property
    def size(self) -> int:
        if self._pool is None:
            return 0
        return self._pool.size

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a task for execution."""
        if self._pool is None:
            raise RuntimeError("WorkerPool not started")

        worker_future = self._pool.submit(fn, *args, **kwargs)
        return _FutureLikeFuture(worker_future)  # type: ignore[return-value]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=wait)

    def __enter__(self) -> IrisWorkerPool:
        self._pool = IrisWorkerPoolImpl(self._client, self._config)
        self._pool.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._pool is not None:
            self._pool.__exit__(*args)


class IrisActorServer:
    """Actor server wrapping Iris ActorServer."""

    def __init__(self, cluster: IrisCluster, host: str = "0.0.0.0", port: int = 0):
        self._cluster = cluster
        self._server = IrisActorServerImpl(host=host, port=port)

    def register(self, name: str, actor: Any) -> None:
        """Register an actor instance."""
        self._server.register(name, actor)

    def serve(self) -> None:
        """Start serving (blocks)."""
        self._server.serve()

    def serve_background(self) -> int:
        """Start serving in background, return port."""
        return self._server.serve_background()

    def shutdown(self) -> None:
        """Shutdown the server."""
        self._server.shutdown()


class IrisCluster:
    """Iris-based cluster implementation for Fray v2.

    Wraps IrisClient to provide the Fray v2 Cluster interface.
    Supports both local and remote modes.
    """

    def __init__(self, client: IrisClient):
        """Initialize with an IrisClient.

        Prefer using factory methods (local(), remote()) over direct construction.

        Args:
            client: IrisClient instance
        """
        self._client = client
        self._jobs: dict[JobId, IrisJob] = {}
        self._lock = threading.Lock()

    @classmethod
    def local(cls, max_workers: int = 4, **kwargs: Any) -> IrisCluster:
        """Create a local IrisCluster.

        Args:
            max_workers: Maximum concurrent jobs
            **kwargs: Additional options (ignored)

        Returns:
            IrisCluster wrapping LocalClusterClient
        """
        config = LocalClientConfig(max_workers=max_workers)
        client = IrisClient.local(config)
        return cls(client)

    @classmethod
    def remote(
        cls,
        controller_address: str,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> IrisCluster:
        """Create a remote IrisCluster.

        Args:
            controller_address: Iris controller URL
            workspace: Path to workspace for bundling
            **kwargs: Additional options (ignored)

        Returns:
            IrisCluster wrapping RemoteClusterClient
        """
        workspace_path = Path(workspace) if workspace else None
        client = IrisClient.remote(controller_address, workspace=workspace_path)
        return cls(client)

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> IrisJob:
        """Submit a job to the cluster."""
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        iris_entrypoint = _v2_to_iris_entrypoint(entrypoint)
        iris_resources = _v2_to_iris_resources(resources)
        iris_environment = _v2_to_iris_environment(environment)

        iris_job = self._client.submit(
            entrypoint=iris_entrypoint,
            name=name,
            resources=iris_resources,
            environment=iris_environment,
        )

        job = IrisJob(iris_job)

        with self._lock:
            self._jobs[job.job_id] = job

        return job

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        self._client.terminate(IrisJobId(str(job_id)))

    def list_jobs(self) -> list[IrisJob]:  # type: ignore[override]
        """List all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def resolver(self) -> IrisResolver:
        """Get resolver for actor discovery."""
        return IrisResolver(self._client.resolver())

    def worker_pool(
        self,
        num_workers: int,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> IrisWorkerPool:
        """Create a worker pool."""
        iris_resources = _v2_to_iris_resources(resources)
        iris_environment = _v2_to_iris_environment(environment)
        return IrisWorkerPool(self._client, num_workers, iris_resources, iris_environment)

    @property
    def namespace(self) -> Namespace:
        # Iris doesn't expose namespace directly, derive from context if available
        from iris.client.client import get_iris_ctx

        ctx = get_iris_ctx()
        if ctx and ctx.job_id:
            return Namespace(str(IrisNamespace.from_job_id(ctx.job_id)))
        return Namespace("iris-default")

    def create_actor_server(self, host: str = "0.0.0.0", port: int = 0) -> IrisActorServer:
        """Create an actor server (internal use by ActorServer)."""
        return IrisActorServer(self, host, port)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the cluster client."""
        self._client.shutdown(wait=wait)

    def __enter__(self) -> IrisCluster:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
