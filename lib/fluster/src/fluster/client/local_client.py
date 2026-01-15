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
from fluster.client.protocols import ResolvedEndpoint, ResolveResult, Resolver
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
    """Endpoint registry implementing the EndpointRegistry protocol.

    Wraps the shared _EndpointStore and auto-prefixes names with the job's
    namespace (derived from context). This class implements the EndpointRegistry
    protocol and is the public API for endpoint registration in local mode.

    The namespace prefix is computed dynamically from the current FlusterContext,
    so a single registry instance can be shared across all jobs.
    """

    def __init__(self, store: _EndpointStore):
        self._store = store

    def _namespace_prefix(self) -> str:
        """Get namespace prefix from current FlusterContext."""
        ctx = get_fluster_ctx()
        if ctx is None:
            raise RuntimeError("No FlusterContext - must be called from within a job")
        return str(Namespace.from_job_id(ctx.job_id))

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint, auto-prefixing with namespace."""
        full_name = f"{self._namespace_prefix()}/{name}"
        return self._store.store(full_name, address, metadata)

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint."""
        self._store.remove(endpoint_id)


class LocalResolver:
    """Resolver backed by _EndpointStore.

    Used by jobs running in LocalClient to discover actors registered
    with the local endpoint store. Auto-prefixes names with namespace.

    The namespace prefix is computed dynamically from the current FlusterContext,
    so a single resolver instance can be shared across all jobs.
    """

    def __init__(self, store: _EndpointStore):
        """Initialize the local resolver.

        Args:
            store: The shared endpoint store to look up from
        """
        self._store = store

    def _namespace_prefix(self) -> str:
        """Get namespace prefix from current FlusterContext."""
        ctx = get_fluster_ctx()
        if ctx is None:
            raise RuntimeError("No FlusterContext - must be called from within a job")
        return str(Namespace.from_job_id(ctx.job_id))

    def resolve(self, name: str) -> ResolveResult:
        """Resolve actor name to endpoints.

        The name is auto-prefixed with the namespace before lookup.

        Args:
            name: Actor name to resolve (will be prefixed)

        Returns:
            ResolveResult with matching endpoints
        """
        prefixed_name = f"{self._namespace_prefix()}/{name}"
        matches = self._store.lookup(prefixed_name)
        endpoints = [ResolvedEndpoint(url=f"http://{addr}", actor_id=eid, metadata=meta) for addr, eid, meta in matches]
        return ResolveResult(name=name, endpoints=endpoints)


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
        self._registry = LocalEndpointRegistry(self._endpoint_store)
        self._resolver = LocalResolver(self._endpoint_store)

    def __enter__(self) -> "LocalClient":
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_workers)
        return self

    def __exit__(self, *_):
        self.shutdown()

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        Uses the namespace derived from the current job context.
        The resolver is shared across all jobs; namespace is looked up
        dynamically from FlusterContext when resolve() is called.
        """
        return self._resolver

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

        # Create context for this job (namespace is derived from job_id root)
        job_ctx = FlusterContext(
            job_id=job_id,
            worker_id=f"local-worker-{threading.current_thread().ident}",
            client=self,
            registry=self._registry,
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
