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

"""LocalClient for running Fluster jobs in threads.

LocalClient provides a way to run Fluster jobs locally without a remote
controller. Jobs execute in threads with proper FlusterContext injection,
enabling the same code to work locally and remotely.

Example:
    from fluster.cluster.local_client import LocalClient, LocalClientConfig
    from fluster.cluster.types import Entrypoint, Namespace
    from fluster import cluster_pb2

    def my_job():
        from fluster.context import fluster_ctx
        ctx = fluster_ctx()
        print(f"Running job {ctx.job_id} in namespace {ctx.namespace}")
        return "success"

    config = LocalClientConfig(max_workers=4, namespace=Namespace("test"))
    with LocalClient(config) as client:
        entrypoint = Entrypoint.from_callable(my_job)
        job_id = client.submit(entrypoint, "test-job", cluster_pb2.ResourceSpec())
        status = client.wait(job_id)
        print(f"Job finished with state: {status.state}")

WorkerPool also works with LocalClient:
    from fluster.worker_pool import WorkerPool, WorkerPoolConfig

    config = LocalClientConfig(max_workers=8, namespace=Namespace("test"))
    with LocalClient(config) as client:
        pool_config = WorkerPoolConfig(num_workers=4, resources=cluster_pb2.ResourceSpec())
        with WorkerPool(client, pool_config) as pool:
            futures = [pool.submit(my_fn, arg) for arg in args]
            results = [f.result() for f in futures]
"""

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from fluster import cluster_pb2
from fluster.actor.resolver import ResolvedEndpoint, Resolver, ResolveResult
from fluster.cluster.types import Entrypoint, JobId, Namespace, is_job_finished
from fluster.context import (
    EndpointRegistry,
    FlusterContext,
    fluster_ctx_scope,
)


@dataclass
class LocalClientConfig:
    """Configuration for local job execution.

    Attributes:
        max_workers: Maximum concurrent job threads
        namespace: Namespace for actor isolation
        port_range: Port range for actor servers (inclusive start, exclusive end)
    """

    max_workers: int = 4
    namespace: Namespace = field(default_factory=lambda: Namespace.DEFAULT)
    port_range: tuple[int, int] = (50000, 60000)


@dataclass
class _LocalJob:
    """Internal job tracking state."""

    job_id: JobId
    future: Future
    state: int = cluster_pb2.JOB_STATE_PENDING
    error: str = ""
    result: Any = None
    started_at_ms: int = 0
    finished_at_ms: int = 0


class LocalEndpointRegistry:
    """In-memory endpoint registry for local execution.

    Thread-safe storage of endpoint registrations, used by ActorServer
    to register endpoints and by LocalResolver to discover them.
    """

    def __init__(self):
        self._endpoints: dict[str, tuple[str, str, dict[str, str]]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint.

        Args:
            name: Actor name for discovery
            address: Address where actor is listening (host:port)
            metadata: Optional metadata

        Returns:
            Unique endpoint ID
        """
        endpoint_id = f"local-ep-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._endpoints[endpoint_id] = (name, address, metadata or {})
        return endpoint_id

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint."""
        with self._lock:
            self._endpoints.pop(endpoint_id, None)

    def lookup(self, name: str) -> list[tuple[str, str, dict[str, str]]]:
        """Look up endpoints by name.

        Args:
            name: Actor name to find

        Returns:
            List of (address, endpoint_id, metadata) tuples
        """
        with self._lock:
            return [
                (addr, eid, meta)
                for eid, (n, addr, meta) in self._endpoints.items()
                if n == name
            ]


class LocalResolver:
    """Resolver backed by LocalEndpointRegistry.

    Used by jobs running in LocalClient to discover actors registered
    with the local endpoint registry.
    """

    def __init__(self, registry: LocalEndpointRegistry, namespace: Namespace):
        self._registry = registry
        self._namespace = namespace

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        """Resolve actor name to endpoints.

        Args:
            name: Actor name to resolve
            namespace: Optional namespace override

        Returns:
            ResolveResult with matching endpoints
        """
        ns = namespace or self._namespace
        matches = self._registry.lookup(name)
        endpoints = [
            ResolvedEndpoint(url=f"http://{addr}", actor_id=eid, metadata=meta)
            for addr, eid, meta in matches
        ]
        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)


class LocalControllerAdapter:
    """ClusterController implementation for LocalClient.

    Provides a consistent interface for job code to interact with the
    local execution environment.
    """

    def __init__(self, client: "LocalClient"):
        self._client = client
        self._registry = LocalEndpointRegistry()

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job for local execution."""
        return self._client.submit(entrypoint, name, resources, environment, ports)

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status."""
        return self._client.status(job_id)

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete."""
        return self._client.wait(job_id, timeout, poll_interval)

    def terminate(self, job_id: JobId) -> None:
        """Terminate a job."""
        self._client.terminate(job_id)

    def resolver(self, namespace: Namespace) -> Resolver:
        """Get a resolver for actor discovery."""
        return LocalResolver(self._registry, namespace)

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        """Get the endpoint registry."""
        return self._registry

    @property
    def address(self) -> str:
        """Address (local client has no network address)."""
        return "local://localhost"


class LocalClient:
    """Run Fluster jobs locally in threads.

    Jobs execute in the current process with proper FlusterContext injection.
    Supports full actor functionality via LocalEndpointRegistry.

    Example:
        config = LocalClientConfig(max_workers=4, namespace=Namespace("test"))
        with LocalClient(config) as client:
            job_id = client.submit(entrypoint, "my-job", resources, ports=["actor"])
            client.wait(job_id)

            # WorkerPool works too!
            pool_config = WorkerPoolConfig(num_workers=3, resources=...)
            with WorkerPool(client, pool_config) as pool:
                future = pool.submit(my_fn, arg)
                result = future.result()
    """

    def __init__(self, config: LocalClientConfig | None = None):
        self._config = config or LocalClientConfig()
        self._executor: ThreadPoolExecutor | None = None
        self._jobs: dict[JobId, _LocalJob] = {}
        self._lock = threading.RLock()
        self._job_counter = 0
        self._next_port = self._config.port_range[0]
        self._controller = LocalControllerAdapter(self)

    def __enter__(self) -> "LocalClient":
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_workers)
        return self

    def __exit__(self, *_):
        self.shutdown()

    @property
    def controller_address(self) -> str:
        """For compatibility with ClusterClient protocol."""
        return "local://localhost"

    def _allocate_port(self) -> int:
        """Allocate a port from the configured range."""
        with self._lock:
            port = self._next_port
            self._next_port += 1
            if self._next_port >= self._config.port_range[1]:
                self._next_port = self._config.port_range[0]
        return port

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job for local execution.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (for logging/debugging)
            resources: Resource requirements (ignored in local mode)
            environment: Environment configuration (env vars applied if provided)
            ports: Port names to allocate (e.g., ["actor"])

        Returns:
            Job ID
        """
        if self._executor is None:
            raise RuntimeError("LocalClient not started. Use 'with LocalClient() as client:' or call __enter__().")

        with self._lock:
            self._job_counter += 1
            job_id = JobId(f"local-{self._job_counter}")

        # Allocate requested ports
        allocated_ports = {port_name: self._allocate_port() for port_name in (ports or [])}

        # Create context for this job
        ctx = FlusterContext(
            namespace=self._config.namespace,
            job_id=job_id,
            worker_id=f"local-worker-{threading.current_thread().ident}",
            controller=self._controller,
            ports=allocated_ports,
        )

        # Create job tracking
        local_job = _LocalJob(
            job_id=job_id,
            future=Future(),
            state=cluster_pb2.JOB_STATE_PENDING,
            started_at_ms=int(time.time() * 1000),
        )

        with self._lock:
            self._jobs[job_id] = local_job

        # Submit to thread pool
        self._executor.submit(self._run_job, local_job, ctx, entrypoint)

        return job_id

    def _run_job(
        self,
        job: _LocalJob,
        ctx: FlusterContext,
        entrypoint: Entrypoint,
    ) -> None:
        """Execute job entrypoint with context injection."""
        job.state = cluster_pb2.JOB_STATE_RUNNING

        try:
            with fluster_ctx_scope(ctx):
                result = entrypoint.callable(*entrypoint.args, **entrypoint.kwargs)
            job.result = result
            job.state = cluster_pb2.JOB_STATE_SUCCEEDED
            job.future.set_result(result)
        except Exception as e:
            job.error = str(e)
            job.state = cluster_pb2.JOB_STATE_FAILED
            job.future.set_exception(e)
        finally:
            job.finished_at_ms = int(time.time() * 1000)

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobStatus proto with current state
        """
        with self._lock:
            job = self._jobs.get(job_id)

        if job is None:
            return cluster_pb2.JobStatus(
                job_id=job_id,
                state=cluster_pb2.JOB_STATE_UNSCHEDULABLE,
            )

        return cluster_pb2.JobStatus(
            job_id=job_id,
            state=job.state,
            error=job.error,
            started_at_ms=job.started_at_ms,
            finished_at_ms=job.finished_at_ms,
        )

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        start = time.time()

        while time.time() - start < timeout:
            status = self.status(job_id)
            if is_job_finished(status.state):
                return status
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Note: In local mode, jobs cannot be forcefully terminated.
        This marks the job as killed but the thread continues until completion.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job and not is_job_finished(job.state):
                job.state = cluster_pb2.JOB_STATE_KILLED
                job.error = "Terminated by user"
                job.finished_at_ms = int(time.time() * 1000)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the local client.

        Args:
            wait: If True, wait for pending jobs to complete
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
