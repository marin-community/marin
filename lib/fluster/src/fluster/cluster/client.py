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

"""Cluster client for job management.

This module provides:
- ClusterClient: Protocol for cluster job operations
- RpcClusterClient: Implementation using RPC to controller
- LocalClient: Implementation for local thread-based execution
- BundleCreator: Helper for creating workspace bundles
"""

import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cloudpickle

from fluster.actor.resolver import ClusterResolver, ResolvedEndpoint, Resolver, ResolveResult
from fluster.time_utils import ExponentialBackoff
from fluster.cluster.types import Entrypoint, JobId, Namespace, is_job_finished
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceClientSync
from fluster.context import (
    EndpointRegistry,
    FlusterContext,
    fluster_ctx_scope,
    get_fluster_ctx,
)


class ClusterClient(Protocol):
    """Protocol for cluster job operations.

    This is the interface WorkerPool and other clients use to interact
    with a cluster. Default implementation is RpcClusterClient.
    """

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job to the cluster.

        The name is combined with the current job's name (if any) to form a
        hierarchical identifier. For example, if the current job is "my-exp"
        and you submit with name="worker-0", the full name becomes "my-exp/worker-0".

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """
        ...

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobInfo proto with current state
        """
        ...

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
            Final JobInfo

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        ...

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Args:
            job_id: Job ID to terminate
        """
        ...

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        The namespace is derived from the current job context.

        Returns:
            Resolver instance
        """
        ...


class BundleCreator:
    """Helper for creating workspace bundles.

    Bundles a user's workspace directory (containing pyproject.toml, uv.lock,
    and source code) into a zip file for job execution.

    The workspace must already have fluster as a dependency in pyproject.toml.
    If uv.lock doesn't exist, it will be generated.
    """

    def __init__(self, workspace: Path):
        """Initialize bundle creator.

        Args:
            workspace: Path to workspace directory containing pyproject.toml
        """
        self._workspace = workspace

    def create_bundle(self) -> bytes:
        """Create a workspace bundle.

        Creates a zip file containing the workspace directory contents.
        Excludes common non-essential files like __pycache__, .git, etc.

        Returns:
            Bundle as bytes (zip file contents)
        """
        with tempfile.TemporaryDirectory(prefix="bundle_") as td:
            bundle_path = Path(td) / "bundle.zip"
            with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in self._workspace.rglob("*"):
                    if file.is_file() and not self._should_exclude(file):
                        zf.write(file, file.relative_to(self._workspace))
            return bundle_path.read_bytes()

    def _should_exclude(self, path: Path) -> bool:
        """Check if a file should be excluded from the bundle."""
        exclude_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "*.pyc",
            "*.egg-info",
            ".venv",
            "venv",
            "node_modules",
        }
        parts = path.relative_to(self._workspace).parts
        for part in parts:
            for pattern in exclude_patterns:
                if pattern.startswith("*"):
                    if part.endswith(pattern[1:]):
                        return True
                elif part == pattern:
                    return True
        return False


class RpcEndpointRegistry:
    """EndpointRegistry implementation that registers via RPC to controller.

    Used by ActorServer to register endpoints when running in a remote worker.
    Names are auto-prefixed with the namespace (root job ID) for isolation.
    """

    def __init__(
        self,
        client: ControllerServiceClientSync,
        job_id: str,
    ):
        """Initialize the RPC endpoint registry.

        Args:
            client: RPC client for controller communication
            job_id: Job ID that owns registered endpoints (namespace derived from root)
        """
        self._client = client
        self._job_id = job_id
        self._namespace_prefix = str(Namespace.from_job_id(job_id))

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint via RPC to controller.

        The name is auto-prefixed with the namespace (root job ID) for isolation.
        For example, registering "calculator" with job "abc123/worker-0" stores
        as "abc123/calculator".

        Args:
            name: Actor name for discovery (will be prefixed)
            address: Address where actor is listening (host:port)
            metadata: Optional metadata

        Returns:
            Endpoint ID assigned by controller
        """
        prefixed_name = f"{self._namespace_prefix}/{name}"
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=prefixed_name,
            address=address,
            job_id=self._job_id,
            metadata=metadata or {},
        )
        response = self._client.register_endpoint(request)
        return response.endpoint_id

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint.

        Not implemented - controller automatically cleans up endpoints when
        jobs terminate.
        """
        raise NotImplementedError(
            "Explicit endpoint unregistration not implemented. "
            "Endpoints are automatically cleaned up when jobs terminate."
        )


class RpcClusterClient:
    """ClusterClient implementation using RPC to controller.

    Can be used in two modes:
    1. External client (with workspace): Bundles workspace and submits jobs from outside.
       Example: client = RpcClusterClient("http://controller:8080", workspace=Path("./my-project"))

    2. Inside-job client (with job_id): Used by job code to submit sub-jobs and register actors.
       Example: client = RpcClusterClient("http://controller:8080", job_id="abc123/worker-0")
    """

    def __init__(
        self,
        controller_address: str,
        *,
        workspace: Path | None = None,
        job_id: str | None = None,
        bundle_gcs_path: str | None = None,
        timeout_ms: int = 30000,
    ):
        """Initialize RPC cluster client.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            workspace: Path to workspace directory containing pyproject.toml.
                If provided, this directory will be bundled and sent to workers.
                Required for external job submission.
            job_id: Current job ID for inside-job use. Enables endpoint_registry
                and parent_job_id inference for sub-job submission.
            bundle_gcs_path: GCS path to workspace bundle for sub-job inheritance.
                When set, sub-jobs use this path instead of creating new bundles.
            timeout_ms: RPC timeout in milliseconds
        """
        self._address = controller_address
        self._workspace = workspace
        self._job_id = job_id
        self._bundle_gcs_path = bundle_gcs_path
        self._timeout_ms = timeout_ms
        self._bundle_blob: bytes | None = None
        self._registry: RpcEndpointRegistry | None = None
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
        )

    def _get_bundle(self) -> bytes:
        """Get workspace bundle (lazy creation with caching)."""
        if self._workspace is None:
            return b""
        if self._bundle_blob is None:
            creator = BundleCreator(self._workspace)
            self._bundle_blob = creator.create_bundle()
        return self._bundle_blob

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        """Get the endpoint registry for actor registration.

        Only available when job_id was provided to constructor.
        """
        if self._registry is None:
            if self._job_id is None:
                raise RuntimeError(
                    "endpoint_registry requires job_id. " "Pass job_id to constructor when using inside a job."
                )
            self._registry = RpcEndpointRegistry(self._client, self._job_id)
        return self._registry

    @property
    def address(self) -> str:
        """Controller address."""
        return self._address

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        The resolver uses the namespace derived from the current job context.
        """
        return ClusterResolver(self._address)

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
    ) -> JobId:
        """Submit a job to the cluster.

        The name is combined with the current job's name (if any) to form a
        hierarchical identifier. For example, if the current job is "my-exp"
        and you submit with name="worker-0", the full name becomes "my-exp/worker-0".

        Child jobs are automatically terminated when their parent is terminated.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])
            scheduling_timeout_seconds: Timeout for scheduling (0 = no timeout)

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """
        # Validate name
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        serialized = cloudpickle.dumps(entrypoint)

        env_config = cluster_pb2.EnvironmentConfig(
            workspace=environment.workspace if environment else "/app",
            pip_packages=list(environment.pip_packages) if environment else [],
            env_vars=dict(environment.env_vars) if environment else {},
            extras=list(environment.extras) if environment else [],
        )

        # Get parent job ID from context or self._job_id
        ctx = get_fluster_ctx()
        parent_job_id = ctx.job_id if ctx else (self._job_id or "")

        # Construct full hierarchical name
        if parent_job_id:
            full_name = f"{parent_job_id}/{name}"
        else:
            full_name = name

        # Use bundle_gcs_path if available (inside-job mode inherits parent workspace),
        # otherwise create bundle_blob from workspace
        if self._bundle_gcs_path:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=full_name,
                serialized_entrypoint=serialized,
                resources=resources,
                environment=env_config,
                bundle_gcs_path=self._bundle_gcs_path,
                ports=ports or [],
                parent_job_id=parent_job_id,
                scheduling_timeout_seconds=scheduling_timeout_seconds,
            )
        else:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=full_name,
                serialized_entrypoint=serialized,
                resources=resources,
                environment=env_config,
                bundle_blob=self._get_bundle(),
                ports=ports or [],
                parent_job_id=parent_job_id,
                scheduling_timeout_seconds=scheduling_timeout_seconds,
            )
        response = self._client.launch_job(request)
        return JobId(response.job_id)

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status."""
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self._client.get_job_status(request)
        return response.job

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete with exponential backoff polling."""
        start = time.monotonic()
        backoff = ExponentialBackoff(initial=0.1, maximum=poll_interval)

        while True:
            job_info = self.status(job_id)
            if is_job_finished(job_info.state):
                return job_info

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            interval = backoff.next_interval()
            remaining = timeout - elapsed
            time.sleep(min(interval, remaining))

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self._client.terminate_job(request)


# =============================================================================
# LocalClient Implementation
# =============================================================================


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


class LocalEndpointRegistry:
    """Shared in-memory endpoint storage for local execution.

    Thread-safe storage of endpoint registrations. Stores full prefixed names.
    Used as shared backing store; per-job prefixing is handled by
    _LocalJobEndpointRegistry wrapper.
    """

    def __init__(self):
        self._endpoints: dict[str, tuple[str, str, dict[str, str]]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        full_name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint with full prefixed name.

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

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint."""
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


class _LocalJobEndpointRegistry:
    """Per-job endpoint registry wrapper that handles namespace prefixing.

    Wraps the shared LocalEndpointRegistry and auto-prefixes names with
    the job's namespace (derived from job_id).
    """

    def __init__(self, shared_registry: LocalEndpointRegistry, job_id: str):
        self._registry = shared_registry
        self._namespace_prefix = str(Namespace.from_job_id(job_id))

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint, auto-prefixing with namespace."""
        full_name = f"{self._namespace_prefix}/{name}"
        return self._registry.register(full_name, address, metadata)

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint."""
        self._registry.unregister(endpoint_id)


class LocalResolver:
    """Resolver backed by LocalEndpointRegistry.

    Used by jobs running in LocalClient to discover actors registered
    with the local endpoint registry. Auto-prefixes names with namespace.
    """

    def __init__(self, registry: LocalEndpointRegistry, job_id: str):
        """Initialize the local resolver.

        Args:
            registry: The shared endpoint registry to look up from
            job_id: Job ID used to derive namespace prefix
        """
        self._registry = registry
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
        matches = self._registry.lookup(prefixed_name)
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
        self._registry = _LocalJobEndpointRegistry(client._shared_registry, job_id)

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
        return LocalResolver(self._client._shared_registry, self._job_id)

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
        self._shared_registry = LocalEndpointRegistry()

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
        return LocalResolver(self._shared_registry, ctx.job_id)

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

        The name is combined with the current job's name (if any) to form a
        hierarchical identifier. For example, if the current job is "my-exp"
        and you submit with name="worker-0", the full name becomes "my-exp/worker-0".

        This matches RpcClusterClient behavior - client constructs the full name.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements (ignored in local mode)
            environment: Environment configuration (env vars applied if provided)
            ports: Port names to allocate (e.g., ["actor"])

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """
        del resources, environment  # Unused in local mode

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
