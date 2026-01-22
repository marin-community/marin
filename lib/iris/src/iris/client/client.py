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
from typing import Protocol

from iris.actor.resolver import ResolvedEndpoint, Resolver, ResolveResult
from iris.cluster.client import (
    BundleCreator,
    JobInfo,
    LocalClusterClient,
    RemoteClusterClient,
    get_job_info,
)
from iris.cluster.types import (
    Constraint,
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    JobId,
    Namespace,
    ResourceSpec,
    is_job_finished,
)
from iris.rpc import cluster_pb2
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)


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


class JobFailedError(Exception):
    """Raised when a job ends in a non-SUCCESS terminal state."""

    def __init__(self, job_id: JobId, status: cluster_pb2.JobStatus):
        self.job_id = job_id
        self.status = status
        state_name = cluster_pb2.JobState.Name(status.state)
        msg = f"Job {job_id} {state_name}"
        if status.error:
            msg += f": {status.error}"
        super().__init__(msg)


@dataclass
class _TaskLogState:
    """Per-task log polling state for multi-task log streaming."""

    task_index: int
    last_timestamp_ms: int = 0


class Task:
    """Handle for a specific task within a job.

    Provides convenient methods for task-level operations like status
    checking and log retrieval.

    Example:
        job = client.submit(entrypoint, "my-job", resources)
        job.wait()
        for task in job.tasks():
            print(f"Task {task.task_index}: {task.state}")
            for entry in task.logs():
                print(entry.data)
    """

    def __init__(self, client: "IrisClient", job_id: JobId, task_index: int):
        self._client = client
        self._job_id = job_id
        self._task_index = task_index

    @property
    def task_index(self) -> int:
        """0-indexed task number within the job."""
        return self._task_index

    @property
    def task_id(self) -> str:
        """Full task identifier (job_id/task-N)."""
        return f"{self._job_id}/task-{self._task_index}"

    @property
    def job_id(self) -> JobId:
        """Parent job identifier."""
        return self._job_id

    def status(self) -> cluster_pb2.TaskStatus:
        """Get current task status.

        Returns:
            TaskStatus proto containing state, worker assignment, and metrics
        """
        return self._client._cluster.get_task_status(str(self._job_id), self._task_index)

    @property
    def state(self) -> cluster_pb2.TaskState:
        """Get current task state (shortcut for status().state)."""
        return self.status().state

    def logs(self, *, start_ms: int = 0, max_lines: int = 0) -> list[LogEntry]:
        """Fetch logs for this task.

        Args:
            start_ms: Only return logs after this timestamp (milliseconds since epoch)
            max_lines: Maximum number of log lines to return (0 = unlimited)

        Returns:
            List of LogEntry objects from the task
        """
        entries = self._client._cluster.fetch_task_logs(self.task_id, start_ms, max_lines)
        return [LogEntry.from_proto(e) for e in entries]


class Job:
    """Handle for a submitted job with convenient methods.

    Returned by IrisClient.submit(). Provides an ergonomic interface for
    common job operations like waiting for completion, checking status,
    and accessing task-level information.

    Example:
        job = client.submit(entrypoint, "my-job", resources)
        status = job.wait()  # Blocks until job completes
        print(f"Job finished: {job.state}")

        for task in job.tasks():
            print(f"Task {task.task_index} logs:")
            for entry in task.logs():
                print(entry.data)
    """

    def __init__(self, client: "IrisClient", job_id: JobId):
        self._client = client
        self._job_id = job_id

    @property
    def job_id(self) -> JobId:
        """Unique job identifier."""
        return self._job_id

    def __str__(self) -> str:
        return str(self._job_id)

    def __repr__(self) -> str:
        return f"Job({self._job_id!r})"

    def status(self) -> cluster_pb2.JobStatus:
        """Get current job status.

        Returns:
            JobStatus proto with current state, task counts, and error info
        """
        return self._client._cluster.get_job_status(self._job_id)

    @property
    def state(self) -> cluster_pb2.JobState:
        """Get current job state (shortcut for status().state)."""
        return self.status().state

    def tasks(self) -> list[Task]:
        """Get all tasks for this job.

        Returns:
            List of Task handles, one per task in the job
        """
        task_statuses = self._client._cluster.list_tasks(str(self._job_id))
        return [Task(self._client, self._job_id, ts.task_index) for ts in task_statuses]

    def wait(
        self,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
        *,
        raise_on_failure: bool = True,
        stream_logs: bool = False,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete.

        Args:
            timeout: Maximum wait time in seconds
            poll_interval: Maximum time between status checks
            raise_on_failure: If True, raise JobFailedError on any non-SUCCESS terminal state
            stream_logs: If True, stream logs from all tasks interleaved

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: Job didn't complete in time
            JobFailedError: Job ended in non-SUCCESS state and raise_on_failure=True
        """
        if not stream_logs:
            status = self._client._cluster.wait_for_job(self._job_id, timeout, poll_interval)
        else:
            status = self._wait_with_multi_task_streaming(timeout, poll_interval)

        if raise_on_failure and status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
            raise JobFailedError(self._job_id, status)

        return status

    def _wait_with_multi_task_streaming(
        self,
        timeout: float,
        poll_interval: float,
    ) -> cluster_pb2.JobStatus:
        """Wait while streaming logs from all tasks."""
        log_states: list[_TaskLogState] = []
        backoff = ExponentialBackoff(initial=0.1, maximum=poll_interval)
        start = time.monotonic()

        while True:
            status = self._client._cluster.get_job_status(self._job_id)

            # Initialize log pollers when we learn num_tasks
            if not log_states and status.num_tasks > 0:
                log_states = [_TaskLogState(i) for i in range(status.num_tasks)]

            # Poll logs from all tasks
            for state in log_states:
                try:
                    task_id = f"{self._job_id}/task-{state.task_index}"
                    entries = self._client._cluster.fetch_task_logs(task_id, state.last_timestamp_ms, 0)
                    for proto in entries:
                        entry = LogEntry.from_proto(proto)
                        state.last_timestamp_ms = max(state.last_timestamp_ms, entry.timestamp_ms)
                        _print_log_entry(self._job_id, entry, task_index=state.task_index)
                except ValueError:
                    logger.debug("Task %d not yet scheduled for job %s", state.task_index, self._job_id)

            if is_job_finished(status.state):
                # Final drain to catch any remaining logs
                for state in log_states:
                    try:
                        task_id = f"{self._job_id}/task-{state.task_index}"
                        entries = self._client._cluster.fetch_task_logs(task_id, state.last_timestamp_ms, 0)
                        for proto in entries:
                            entry = LogEntry.from_proto(proto)
                            _print_log_entry(self._job_id, entry, task_index=state.task_index)
                    except ValueError:
                        logger.debug(
                            "Task %d logs unavailable during final drain for job %s", state.task_index, self._job_id
                        )
                return status

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Job {self._job_id} did not complete in {timeout}s")

            time.sleep(backoff.next_interval())

    def terminate(self) -> None:
        """Terminate this job."""
        self._client._cluster.terminate_job(self._job_id)


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
    client: "IrisClient | None" = None
    registry: "NamespacedEndpointRegistry | None" = None
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
        client: "IrisClient | None" = None,
        registry: "NamespacedEndpointRegistry | None" = None,
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


@dataclass
class LocalClientConfig:
    """Configuration for local job execution.

    Attributes:
        max_workers: Maximum concurrent job threads
        port_range: Port range for actor servers (inclusive start, exclusive end)
    """

    max_workers: int = 4
    port_range: tuple[int, int] = (50000, 60000)


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


class NamespacedResolver:
    """Resolver that auto-prefixes names with namespace.

    Can be used in two modes:
    1. With explicit namespace: Use directly from client code without IrisContext
    2. Without explicit namespace: Derives namespace from IrisContext (requires job context)
    """

    def __init__(
        self,
        cluster: LocalClusterClient | RemoteClusterClient,
        namespace: Namespace | None = None,
    ):
        self._cluster = cluster
        self._explicit_namespace = namespace

    def _namespace_prefix(self) -> str:
        if self._explicit_namespace is not None:
            return str(self._explicit_namespace)
        ctx = get_iris_ctx()
        if ctx is None:
            raise RuntimeError("No IrisContext - provide explicit namespace or call from within a job")
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


class IrisClient:
    """High-level client with automatic job hierarchy and namespace-based actor discovery.

    Example:
        # Local execution
        with IrisClient.local() as client:
            job = client.submit(entrypoint, "my-job", resources)
            job.wait()

        # Remote execution
        client = IrisClient.remote("http://controller:8080", workspace=Path("."))
        job = client.submit(entrypoint, "my-job", resources)
        status = job.wait()
        for task in job.tasks():
            for entry in task.logs():
                print(entry.data)
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

    def resolver_for_job(self, job_id: JobId | str) -> Resolver:
        """Get a resolver for endpoints registered by a specific job.

        Use this when resolving endpoints from outside a job context, such as
        from WorkerPool which runs in client context but needs to resolve
        endpoints registered by its worker jobs.

        Args:
            job_id: The job whose namespace to resolve endpoints in

        Returns:
            Resolver that prefixes lookups with the job's namespace
        """
        namespace = Namespace.from_job_id(str(job_id))
        return NamespacedResolver(self._cluster, namespace=namespace)

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
        constraints: list[Constraint] | None = None,
        coscheduling: CoschedulingConfig | None = None,
    ) -> Job:
        """Submit a job with automatic job_id hierarchy.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])
            scheduling_timeout_seconds: Maximum time to wait for scheduling (0 = no timeout)
            constraints: Constraints for filtering workers by attribute
            coscheduling: Configuration for atomic multi-task scheduling

        Returns:
            Job handle for the submitted job

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

        # Convert to wire format
        resources_proto = resources.to_proto()
        environment_proto = environment.to_proto() if environment else None
        constraints_proto = [c.to_proto() for c in constraints or []]
        coscheduling_proto = coscheduling.to_proto() if coscheduling else None

        self._cluster.submit_job(
            job_id=job_id,
            entrypoint=entrypoint,
            resources=resources_proto,
            environment=environment_proto,
            ports=ports,
            scheduling_timeout_seconds=scheduling_timeout_seconds,
            constraints=constraints_proto,
            coscheduling=coscheduling_proto,
        )

        return Job(self, job_id)

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobStatus proto with current state
        """
        return self._cluster.get_job_status(job_id)

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

    def task_status(self, job_id: JobId, task_index: int) -> cluster_pb2.TaskStatus:
        """Get status of a specific task within a job.

        Args:
            job_id: Job identifier
            task_index: 0-indexed task number

        Returns:
            TaskStatus proto containing state, worker assignment, and metrics
        """
        return self._cluster.get_task_status(str(job_id), task_index)

    def list_tasks(self, job_id: JobId) -> list[cluster_pb2.TaskStatus]:
        """List all tasks for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of TaskStatus protos, one per task
        """
        return self._cluster.list_tasks(str(job_id))

    def fetch_task_logs(
        self,
        job_id: JobId,
        task_index: int,
        *,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[LogEntry]:
        """Fetch logs for a specific task.

        Args:
            job_id: Job identifier
            task_index: 0-indexed task number
            start_ms: Only return logs after this timestamp (milliseconds since epoch)
            max_lines: Maximum number of log lines to return (0 = unlimited)

        Returns:
            List of LogEntry objects from the task
        """
        task_id = f"{job_id}/task-{task_index}"
        entries = self._cluster.fetch_task_logs(task_id, start_ms, max_lines)
        return [LogEntry.from_proto(e) for e in entries]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the client.

        Args:
            wait: If True, wait for pending jobs to complete (local mode only)
        """
        self._cluster.shutdown(wait=wait)


def _print_log_entry(job_id: JobId, entry: LogEntry, task_index: int | None = None) -> None:
    """Log a job log entry."""
    ts = datetime.fromtimestamp(entry.timestamp_ms / 1000, tz=timezone.utc)
    ts_str = ts.strftime("%H:%M:%S")
    if task_index is not None:
        logger.info("[%s/task-%d][%s] %s", job_id, task_index, ts_str, entry.data)
    else:
        logger.info("[%s][%s] %s", job_id, ts_str, entry.data)


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
