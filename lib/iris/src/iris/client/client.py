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

"""High-level client with automatic job hierarchy and namespace-based actor discovery.

Example:
    # In job code:
    from iris.client import iris_ctx

    ctx = iris_ctx()
    print(f"Running job {ctx.job_id} in namespace {ctx.namespace}")

    # Get allocated port for actor server
    port = ctx.get_port("actor")

    # Submit a sub-job
    sub_job_id = ctx.client.submit(entrypoint, "sub-job", resources)
"""

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from iris.actor.resolver import ResolvedEndpoint, ResolveResult, Resolver
from iris.cluster.client import (
    BundleCreator,
    JobInfo,
    LocalClusterClient,
    RemoteClusterClient,
    get_job_info,
)
from iris.cluster.types import Entrypoint, is_job_finished, JobId, Namespace
from iris.rpc import cluster_pb2
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)

# =============================================================================
# Log Entry
# =============================================================================


@dataclass
class LogEntry:
    """A single log line from a job.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds
        source: Log source - "stdout", "stderr", or "build"
        data: Log line content
    """

    timestamp_ms: int
    source: str
    data: str

    @classmethod
    def from_proto(cls, proto: cluster_pb2.Worker.LogEntry) -> "LogEntry":
        return cls(
            timestamp_ms=proto.timestamp_ms,
            source=proto.source,
            data=proto.data,
        )


# =============================================================================
# Context Management
# =============================================================================


@dataclass
class IrisContext:
    """Unified execution context for Iris.

    Available in any iris job via `iris_ctx()`. Contains all
    information about the current execution environment.

    The namespace is derived from the job_id: all jobs in a hierarchy share the
    same namespace (the root job's ID). This makes namespace an actor-only concept
    that doesn't need to be explicitly passed around.

    Attributes:
        job_id: Unique identifier for this job (hierarchical: "root/parent/child")
        attempt_id: Attempt number for this job execution (0-based)
        worker_id: Identifier for the worker executing this job (may be None)
        client: IrisClient for job operations (submit, status, wait, etc.)
        registry: EndpointRegistry for actor endpoint registration
        ports: Allocated ports by name (e.g., {"actor": 50001})
    """

    job_id: str
    attempt_id: int = 0
    worker_id: str | None = None
    client: Any = None  # IrisClient (can't type due to forward ref)
    registry: Any = None  # NamespacedEndpointRegistry (can't type due to forward ref)
    ports: dict[str, int] | None = None

    def __post_init__(self):
        if self.ports is None:
            self.ports = {}

    @property
    def namespace(self) -> Namespace:
        """Namespace derived from the root job ID.

        All jobs in a hierarchy share the same namespace, enabling actors
        to be discovered across the job tree.
        """
        return Namespace.from_job_id(self.job_id)

    @property
    def parent_job_id(self) -> str | None:
        """Parent job ID, or None if this is a root job.

        For job_id "root/parent/child", returns "root/parent".
        For job_id "root", returns None.
        """
        parts = self.job_id.rsplit("/", 1)
        if len(parts) == 1:
            return None
        return parts[0]

    def get_port(self, name: str) -> int:
        """Get an allocated port by name.

        Args:
            name: Port name (e.g., "actor")

        Returns:
            Port number

        Raises:
            KeyError: If port was not allocated for this job
        """
        if self.ports is None or name not in self.ports:
            available = list(self.ports.keys()) if self.ports else []
            raise KeyError(
                f"Port '{name}' not allocated. "
                f"Available ports: {available or 'none'}. "
                f"Did you request ports=['actor'] when submitting the job?"
            )
        return self.ports[name]

    @property
    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        The resolver uses the namespace derived from this context's job ID.

        Raises:
            RuntimeError: If no client is available
        """
        if self.client is None:
            raise RuntimeError("No client available in context")
        return self.client.resolver()

    @staticmethod
    def from_job_info(
        info: JobInfo,
        client: Any = None,
        registry: Any = None,
    ) -> "IrisContext":
        """Create IrisContext from JobInfo.

        Args:
            info: JobInfo from cluster layer
            client: Optional IrisClient instance
            registry: Optional EndpointRegistry instance

        Returns:
            IrisContext with metadata from JobInfo
        """
        return IrisContext(
            job_id=info.job_id,
            attempt_id=info.attempt_id,
            worker_id=info.worker_id,
            client=client,
            registry=registry,
            ports=dict(info.ports),
        )


# Module-level ContextVar for the current iris context
_iris_context: ContextVar[IrisContext | None] = ContextVar(
    "iris_context",
    default=None,
)


def iris_ctx() -> IrisContext:
    """Get or create IrisContext from environment.

    Returns:
        Current IrisContext
    """
    ctx = _iris_context.get()
    if ctx is None:
        ctx = create_context_from_env()
        _iris_context.set(ctx)
    return ctx


def get_iris_ctx() -> IrisContext | None:
    """Get the current IrisContext, or None if not in a job.

    Unlike iris_ctx(), this function does not auto-create context
    from environment if called outside a job context.

    Returns:
        Current IrisContext or None
    """
    return _iris_context.get()


@contextmanager
def iris_ctx_scope(ctx: IrisContext) -> Generator[IrisContext, None, None]:
    """Set the iris context for the duration of this scope.

    Args:
        ctx: Context to set for this scope

    Yields:
        The provided context

    Example:
        ctx = IrisContext(job_id="my-namespace/job-1", worker_id="worker-1")
        with iris_ctx_scope(ctx):
            my_job_function()
    """
    token = _iris_context.set(ctx)
    try:
        yield ctx
    finally:
        _iris_context.reset(token)


# =============================================================================
# Client Configuration
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


# =============================================================================
# Endpoint Registry
# =============================================================================


class EndpointRegistry(Protocol):
    """Protocol for registering actor endpoints with automatic namespace prefixing."""

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint for actor discovery.

        Args:
            name: Actor name for discovery
            address: Address where actor is listening (host:port)
            metadata: Optional metadata for the endpoint

        Returns:
            Unique endpoint ID for later unregistration
        """
        ...

    def unregister(self, endpoint_id: str) -> None:
        """Unregister a previously registered endpoint.

        Args:
            endpoint_id: ID returned from register()
        """
        ...


class NamespacedEndpointRegistry:
    """Endpoint registry that auto-prefixes names with namespace from IrisContext."""

    def __init__(self, cluster: LocalClusterClient | RemoteClusterClient):
        self._cluster = cluster

    def _get_context(self) -> tuple[str, str]:
        """Get job_id and namespace_prefix from current IrisContext."""
        ctx = get_iris_ctx()
        if ctx is None:
            raise RuntimeError("No IrisContext - must be called from within a job")
        namespace_prefix = str(Namespace.from_job_id(ctx.job_id))
        return ctx.job_id, namespace_prefix

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint, auto-prefixing with namespace.

        Args:
            name: Actor name for discovery (will be prefixed)
            address: Address where actor is listening (host:port)
            metadata: Optional metadata

        Returns:
            Endpoint ID
        """
        job_id, namespace_prefix = self._get_context()
        prefixed_name = f"{namespace_prefix}/{name}"
        return self._cluster.register_endpoint(
            name=prefixed_name,
            address=address,
            job_id=job_id,
            metadata=metadata,
        )

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint.

        Args:
            endpoint_id: Endpoint ID to remove
        """
        self._cluster.unregister_endpoint(endpoint_id)


# =============================================================================
# Namespace-aware Resolver
# =============================================================================


class NamespacedResolver:
    """Resolver that auto-prefixes names with namespace from IrisContext."""

    def __init__(self, cluster: LocalClusterClient | RemoteClusterClient):
        self._cluster = cluster

    def _namespace_prefix(self) -> str:
        ctx = get_iris_ctx()
        if ctx is None:
            raise RuntimeError("No IrisContext - must be called from within a job")
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
        matches = self._cluster.list_endpoints(prefix=prefixed_name)

        # Filter to exact matches
        endpoints = [
            ResolvedEndpoint(
                url=f"http://{ep.address}",
                actor_id=ep.endpoint_id,
                metadata=dict(ep.metadata),
            )
            for ep in matches
            if ep.name == prefixed_name
        ]

        return ResolveResult(name=name, endpoints=endpoints)


# =============================================================================
# IrisClient
# =============================================================================


class IrisClient:
    """High-level client with automatic job hierarchy and namespace-based actor discovery.

    Example:
        # Local execution
        with IrisClient.local() as client:
            job_id = client.submit(entrypoint, "my-job", resources)
            client.wait(job_id)

        # Remote execution
        client = IrisClient.remote("http://controller:8080", workspace=Path("."))
        job_id = client.submit(entrypoint, "my-job", resources)
        client.wait(job_id)
    """

    def __init__(self, cluster: LocalClusterClient | RemoteClusterClient):
        """Initialize IrisClient with a cluster client.

        Prefer using factory methods (local(), remote()) over direct construction.

        Args:
            cluster: Low-level cluster client (LocalClusterClient or RemoteClusterClient)
        """
        self._cluster = cluster
        self._registry = NamespacedEndpointRegistry(cluster)
        self._resolver = NamespacedResolver(cluster)

    @classmethod
    def local(cls, config: LocalClientConfig | None = None) -> "IrisClient":
        """Create an IrisClient for local execution using real Controller/Worker.

        Args:
            config: Configuration for local execution

        Returns:
            IrisClient wrapping LocalClusterClient
        """
        cfg = config or LocalClientConfig()
        cluster = LocalClusterClient.create(
            max_workers=cfg.max_workers,
            port_range=cfg.port_range,
        )
        return cls(cluster)

    @classmethod
    def remote(
        cls,
        controller_address: str,
        *,
        workspace: Path | None = None,
        bundle_gcs_path: str | None = None,
        timeout_ms: int = 30000,
    ) -> "IrisClient":
        """Create an IrisClient for RPC-based cluster execution.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            workspace: Path to workspace directory containing pyproject.toml.
                If provided, this directory will be bundled and sent to workers.
                Required for external job submission.
            bundle_gcs_path: GCS path to workspace bundle for sub-job inheritance.
                When set, sub-jobs use this path instead of creating new bundles.
            timeout_ms: RPC timeout in milliseconds

        Returns:
            IrisClient wrapping RemoteClusterClient
        """
        bundle_blob = None
        if workspace is not None:
            creator = BundleCreator(workspace)
            bundle_blob = creator.create_bundle()

        cluster = RemoteClusterClient(
            controller_address=controller_address,
            bundle_gcs_path=bundle_gcs_path,
            bundle_blob=bundle_blob,
            timeout_ms=timeout_ms,
        )
        return cls(cluster)

    def __enter__(self) -> "IrisClient":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        return self._registry

    def resolver(self) -> Resolver:
        return self._resolver

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpecProto,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
    ) -> JobId:
        """Submit a job with automatic job_id hierarchy.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])
            scheduling_timeout_seconds: Maximum time to wait for scheduling (0 = no timeout)

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        # Get parent job ID from context
        ctx = get_iris_ctx()
        parent_job_id = ctx.job_id if ctx else None

        # Construct full hierarchical name
        if parent_job_id:
            job_id = JobId(f"{parent_job_id}/{name}")
        else:
            job_id = JobId(name)

        self._cluster.submit_job(
            job_id=job_id,
            entrypoint=entrypoint,
            resources=resources,
            environment=environment,
            ports=ports,
            scheduling_timeout_seconds=scheduling_timeout_seconds,
        )

        return job_id

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobStatus proto with current state
        """
        return self._cluster.get_job_status(job_id)

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
        *,
        stream_logs: bool = False,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Maximum time between status checks
            stream_logs: If True, stream logs to stdout while waiting

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        if not stream_logs:
            return self._cluster.wait_for_job(job_id, timeout, poll_interval)

        last_timestamp_ms = 0
        start = time.monotonic()
        backoff = ExponentialBackoff(initial=0.1, maximum=poll_interval)

        while True:
            # Fetch and emit logs since last check
            try:
                logs = self.fetch_logs(job_id, start_ms=last_timestamp_ms)
                for entry in logs:
                    last_timestamp_ms = max(last_timestamp_ms, entry.timestamp_ms)
                    _print_log_entry(job_id, entry)
            except ValueError:
                pass  # Job not yet scheduled

            # Check job status
            status = self._cluster.get_job_status(job_id)
            if is_job_finished(status.state):
                # Final drain to catch any remaining logs
                try:
                    for entry in self.fetch_logs(job_id, start_ms=last_timestamp_ms):
                        _print_log_entry(job_id, entry)
                except ValueError:
                    pass
                return status

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            time.sleep(backoff.next_interval())

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Args:
            job_id: Job ID to terminate
        """
        self._cluster.terminate_job(job_id)

    def list_jobs(
        self,
        *,
        states: list[cluster_pb2.JobState] | None = None,
        prefix: str | None = None,
    ) -> list[cluster_pb2.JobStatus]:
        """List jobs with optional filtering.

        Args:
            states: If provided, only return jobs in these states
            prefix: If provided, only return jobs whose job_id starts with this prefix

        Returns:
            List of JobStatus matching the filters
        """
        all_jobs = self._cluster.list_jobs()
        result = []
        for job in all_jobs:
            if states is not None and job.state not in states:
                continue
            if prefix is not None and not job.job_id.startswith(prefix):
                continue
            result.append(job)
        return result

    def terminate_prefix(
        self,
        prefix: str,
        *,
        exclude_finished: bool = True,
    ) -> list[JobId]:
        """Terminate all jobs matching a prefix.

        Args:
            prefix: Job ID prefix to match (e.g., "my-experiment/")
            exclude_finished: If True, skip jobs already in terminal states

        Returns:
            List of job IDs that were terminated
        """
        terminal_states = {
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        }

        jobs = self.list_jobs(prefix=prefix)
        terminated = []
        for job in jobs:
            if exclude_finished and job.state in terminal_states:
                continue
            self.terminate(JobId(job.job_id))
            terminated.append(JobId(job.job_id))
        return terminated

    def fetch_logs(
        self,
        job_id: JobId,
        *,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[LogEntry]:
        """Fetch logs for a job.

        Args:
            job_id: Job ID to fetch logs for
            start_ms: Only return logs after this timestamp (exclusive, for incremental polling)
            max_lines: Maximum number of lines to return (0 = unlimited)

        Returns:
            List of LogEntry objects
        """
        log_protos = self._cluster.fetch_logs(job_id, start_ms=start_ms, max_lines=max_lines)
        return [LogEntry.from_proto(p) for p in log_protos]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the client.

        Args:
            wait: If True, wait for pending jobs to complete (local mode only)
        """
        self._cluster.shutdown(wait=wait)


def _print_log_entry(job_id: JobId, entry: LogEntry) -> None:
    """Log a job log entry."""
    ts = datetime.fromtimestamp(entry.timestamp_ms / 1000, tz=timezone.utc)
    ts_str = ts.strftime("%H:%M:%S")
    logger.info("[%s][%s] %s", job_id, ts_str, entry.data)


# =============================================================================
# Context Creation from Environment
# =============================================================================


def create_context_from_env() -> IrisContext:
    """Create IrisContext from IRIS_* environment variables.

    Returns:
        Configured IrisContext
    """
    # Get job info from environment
    job_info = get_job_info()
    if job_info is None:
        # If no job info available, create minimal context
        return IrisContext(job_id="")

    # Set up client and registry if controller address is available
    client = None
    registry = None
    if job_info.controller_address:
        bundle_gcs_path = os.environ.get("IRIS_BUNDLE_GCS_PATH")

        # Create remote client for context use
        client = IrisClient.remote(
            controller_address=job_info.controller_address,
            bundle_gcs_path=bundle_gcs_path,
        )

        # Use the client's internal registry which handles namespace prefixing
        registry = client._registry

    return IrisContext.from_job_info(job_info, client=client, registry=registry)
