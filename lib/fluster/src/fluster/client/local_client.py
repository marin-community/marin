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

"""Local execution client for Fluster.

This module provides LocalClient, which runs Fluster jobs locally in threads
rather than on a remote cluster. Useful for development and testing.

Classes:
- LocalClientConfig: Configuration for local execution
- _EndpointStore: Internal shared in-memory endpoint storage
- LocalEndpointRegistry: Per-job endpoint registry (implements EndpointRegistry protocol)
- LocalResolver: Resolver backed by _EndpointStore
- LocalClient: Thread-based job execution client
"""

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from fluster.client.context import FlusterContext, fluster_ctx_scope, get_fluster_ctx
from fluster.client.protocols import EndpointRegistry, ResolvedEndpoint, ResolveResult, Resolver
from fluster.cluster.types import Entrypoint, JobId, Namespace, is_job_finished
from fluster.rpc import cluster_pb2
from fluster.time_utils import ExponentialBackoff


@dataclass
class LocalClientConfig:
    """Configuration for local job execution.

    Attributes:
        max_workers: Maximum concurrent job threads
        port_range: Port range for actor servers (inclusive start, exclusive end)
    """

    max_workers: int = 4
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


class _EndpointStore:
    """Internal shared in-memory endpoint storage for local execution.

    Thread-safe storage of endpoint registrations. Stores full prefixed names.
    Used as shared backing store; per-job namespace prefixing is handled by
    LocalEndpointRegistry wrapper which implements the EndpointRegistry protocol.

    This class is internal and does NOT implement the EndpointRegistry protocol.
    """

    def __init__(self):
        self._endpoints: dict[str, tuple[str, str, dict[str, str]]] = {}
        self._lock = threading.RLock()

    def store(
        self,
        full_name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store an endpoint with full prefixed name.

        Args:
            full_name: Full prefixed name (e.g., "abc123/calculator")
            address: Address where actor is listening (host:port)
            metadata: Optional metadata

        Returns:
            Unique endpoint ID
        """
        endpoint_id = f"local-ep-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._endpoints[endpoint_id] = (full_name, address, metadata or {})
        return endpoint_id

    def remove(self, endpoint_id: str) -> None:
        """Remove an endpoint from storage."""
        with self._lock:
            self._endpoints.pop(endpoint_id, None)

    def lookup(self, full_name: str) -> list[tuple[str, str, dict[str, str]]]:
        """Look up endpoints by full prefixed name.

        Args:
            full_name: Full prefixed name to find (e.g., "abc123/calculator")

        Returns:
            List of (address, endpoint_id, metadata) tuples
        """
        with self._lock:
            return [(addr, eid, meta) for eid, (n, addr, meta) in self._endpoints.items() if n == full_name]


class LocalEndpointRegistry:
    """Per-job endpoint registry implementing the EndpointRegistry protocol.

    Wraps the shared _EndpointStore and auto-prefixes names with the job's
    namespace (derived from job_id). This class implements the EndpointRegistry
    protocol and is the public API for endpoint registration in local mode.
    """

    def __init__(self, store: _EndpointStore, job_id: str):
        self._store = store
        self._namespace_prefix = str(Namespace.from_job_id(job_id))

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint, auto-prefixing with namespace."""
        full_name = f"{self._namespace_prefix}/{name}"
        return self._store.store(full_name, address, metadata)

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint."""
        self._store.remove(endpoint_id)


class LocalResolver:
    """Resolver backed by _EndpointStore.

    Used by jobs running in LocalClient to discover actors registered
    with the local endpoint store. Auto-prefixes names with namespace.
    """

    def __init__(self, store: _EndpointStore, job_id: str):
        """Initialize the local resolver.

        Args:
            store: The shared endpoint store to look up from
            job_id: Job ID used to derive namespace prefix
        """
        self._store = store
        self._namespace_prefix = str(Namespace.from_job_id(job_id))

    def resolve(self, name: str) -> ResolveResult:
        """Resolve actor name to endpoints.

        The name is auto-prefixed with the namespace before lookup.

        Args:
            name: Actor name to resolve (will be prefixed)

        Returns:
            ResolveResult with matching endpoints
        """
        prefixed_name = f"{self._namespace_prefix}/{name}"
        matches = self._store.lookup(prefixed_name)
        endpoints = [ResolvedEndpoint(url=f"http://{addr}", actor_id=eid, metadata=meta) for addr, eid, meta in matches]
        return ResolveResult(name=name, endpoints=endpoints)


class _LocalJobControllerAdapter:
    """Per-job ClusterController implementation for LocalClient.

    Each job gets its own adapter instance with its job_id, enabling
    proper namespace derivation that matches RPC behavior.
    """

    def __init__(self, client: "LocalClient", job_id: str):
        self._client = client
        self._job_id = job_id
        self._registry = LocalEndpointRegistry(client._endpoint_store, job_id)

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

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery."""
        return LocalResolver(self._client._endpoint_store, self._job_id)

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
    Supports full actor functionality via _EndpointStore and LocalEndpointRegistry.

    Example:
        config = LocalClientConfig(max_workers=4)
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
        self._endpoint_store = _EndpointStore()

    def __enter__(self) -> "LocalClient":
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_workers)
        return self

    def __exit__(self, *_):
        self.shutdown()

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        Uses the namespace derived from the current job context.
        Must be called from within a job.
        """
        ctx = get_fluster_ctx()
        if ctx is None:
            raise RuntimeError(
                "LocalClient.resolver() requires a FlusterContext. " "Call this from within a submitted job."
            )
        return LocalResolver(self._endpoint_store, ctx.job_id)

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
        _resources: cluster_pb2.ResourceSpec,
        _environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job for local execution.

        The name is combined with the current job's name (if any) to form a
        hierarchical identifier. For example, if the current job is "my-exp"
        and you submit with name="worker-0", the full name becomes "my-exp/worker-0".

        This matches RpcClusterClient behavior - client constructs the full name.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            _resources: Resource requirements (ignored in local mode)
            _environment: Environment configuration (ignored in local mode)
            ports: Port names to allocate (e.g., ["actor"])

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """

        # Validate name
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        if self._executor is None:
            raise RuntimeError("LocalClient not started. Use 'with LocalClient() as client:' or call __enter__().")

        # Get parent job ID from context
        ctx = get_fluster_ctx()
        parent_job_id = ctx.job_id if ctx else None

        # Construct full hierarchical name (matches RpcClusterClient)
        if parent_job_id:
            job_id = JobId(f"{parent_job_id}/{name}")
        else:
            job_id = JobId(name)

        # Allocate requested ports
        allocated_ports = {port_name: self._allocate_port() for port_name in ports or []}

        # Create per-job controller adapter (matches RPC behavior)
        job_controller = _LocalJobControllerAdapter(self, job_id)

        # Create context for this job (namespace is derived from job_id root)
        job_ctx = FlusterContext(
            job_id=job_id,
            worker_id=f"local-worker-{threading.current_thread().ident}",
            controller=job_controller,
            ports=allocated_ports,
        )

        # Reject duplicates (must check under lock for thread safety)
        with self._lock:
            if job_id in self._jobs:
                raise ValueError(f"Job {job_id} already exists")

            # Create job tracking
            local_job = _LocalJob(
                job_id=job_id,
                future=Future(),
                state=cluster_pb2.JOB_STATE_PENDING,
                started_at_ms=int(time.time() * 1000),
            )
            self._jobs[job_id] = local_job

        # Submit to thread pool
        self._executor.submit(self._run_job, local_job, job_ctx, entrypoint)

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
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete with exponential backoff polling.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Maximum time between status checks

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        start = time.monotonic()
        backoff = ExponentialBackoff(initial=0.1, maximum=poll_interval)

        while True:
            status = self.status(job_id)
            if is_job_finished(status.state):
                return status

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            interval = backoff.next_interval()
            remaining = timeout - elapsed
            time.sleep(min(interval, remaining))

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
