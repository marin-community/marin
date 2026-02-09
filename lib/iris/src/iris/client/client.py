# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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
    JobName,
    Namespace,
    ResourceSpec,
    is_job_finished,
)
from iris.rpc import cluster_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


@dataclass
class TaskLogEntry:
    """A log entry with task context.

    Attributes:
        timestamp: When the log line was produced
        worker_id: Worker that produced this log
        task_id: Task that produced this log
        source: Log source - "stdout", "stderr", or "build"
        data: Log line content
        attempt_id: Which attempt produced this log (0-indexed)
    """

    timestamp: Timestamp
    worker_id: str
    task_id: JobName
    source: str
    data: str
    attempt_id: int = 0


@dataclass
class TaskLogError:
    """Error fetching logs for a task.

    Attributes:
        task_id: Task we failed to fetch logs for
        worker_id: Worker we tried to contact (may be empty if task unassigned)
        error: Error message
    """

    task_id: JobName
    worker_id: str
    error: str


@dataclass
class TaskLogsResult:
    """Result of fetching task logs.

    Attributes:
        entries: Log entries sorted by timestamp
        errors: Errors encountered while fetching logs
        last_timestamp_ms: Maximum timestamp seen (for pagination cursor)
        truncated: Whether results were truncated due to max_lines limit
    """

    entries: list[TaskLogEntry]
    errors: list[TaskLogError]
    last_timestamp_ms: int
    truncated: bool


def _log_task_results(result: TaskLogsResult) -> None:
    """Log task results to the logger, including any errors."""
    for error in result.errors:
        logger.warning(
            "task=%s worker=%s | error fetching logs: %s",
            error.task_id,
            error.worker_id or "?",
            error.error,
        )
    for entry in result.entries:
        logger.info(
            "worker=%s task=%s attempt=%d | %s",
            entry.worker_id,
            entry.task_id,
            entry.attempt_id,
            entry.data,
        )


class JobFailedError(Exception):
    """Raised when a job ends in a non-SUCCESS terminal state."""

    def __init__(self, job_id: JobName, status: cluster_pb2.JobStatus):
        self.job_id = job_id
        self.status = status
        state_name = cluster_pb2.JobState.Name(status.state)
        msg = f"Job {job_id} {state_name}"
        if status.error:
            msg += f": {status.error}"
        super().__init__(msg)


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

    def __init__(self, client: "IrisClient", task_name: JobName):
        self._client = client
        self._task_name = task_name

    @property
    def task_index(self) -> int:
        """0-indexed task number within the job."""
        return self._task_name.require_task()[1]

    @property
    def task_id(self) -> JobName:
        """Full task identifier (/job/.../index)."""
        return self._task_name

    @property
    def job_id(self) -> JobName:
        """Parent job identifier."""
        return self._task_name.parent or self._task_name

    def status(self) -> cluster_pb2.TaskStatus:
        """Get current task status.

        Returns:
            TaskStatus proto containing state, worker assignment, and metrics
        """
        return self._client._cluster_client.get_task_status(self.task_id)

    @property
    def state(self) -> cluster_pb2.TaskState:
        """Get current task state (shortcut for status().state)."""
        return self.status().state

    def logs(self, *, start: Timestamp | None = None, max_lines: int = 0) -> list[TaskLogEntry]:
        """Fetch logs for this task.

        Args:
            start: Only return logs after this timestamp (None = from beginning)
            max_lines: Maximum number of log lines to return (0 = unlimited)

        Returns:
            List of TaskLogEntry objects from the task
        """
        response = self._client._cluster_client.fetch_task_logs(
            self.task_id,
            since_ms=start.epoch_ms() if start else 0,
            max_total_lines=max_lines,
        )
        if response.task_logs:
            batch = response.task_logs[0]
            return [
                TaskLogEntry(
                    timestamp=Timestamp.from_proto(e.timestamp),
                    worker_id=batch.worker_id or "",
                    task_id=self.task_id,
                    source=e.source,
                    data=e.data,
                    attempt_id=e.attempt_id,
                )
                for e in batch.logs
            ]
        return []


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

    def __init__(self, client: "IrisClient", job_id: JobName):
        self._client = client
        self._job_id = job_id

    @property
    def job_id(self) -> JobName:
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
        return self._client._cluster_client.get_job_status(self._job_id)

    @property
    def state(self) -> cluster_pb2.JobState:
        """Get current job state (shortcut for status().state)."""
        return self.status().state

    def tasks(self) -> list[Task]:
        """Get all tasks for this job.

        Returns:
            List of Task handles, one per task in the job
        """
        task_statuses = self._client._cluster_client.list_tasks(self._job_id)
        return [Task(self._client, JobName.from_wire(ts.task_id)) for ts in task_statuses]

    def wait(
        self,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
        *,
        raise_on_failure: bool = True,
        stream_logs: bool = False,
        include_children: bool = False,
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
            status = self._client._cluster_client.wait_for_job(self._job_id, timeout, poll_interval)
        else:
            status = self._wait_with_multi_task_streaming(timeout, poll_interval, include_children)

        if raise_on_failure and status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
            raise JobFailedError(self._job_id, status)

        return status

    def _wait_with_multi_task_streaming(
        self,
        timeout: float,
        poll_interval: float,
        include_children: bool,
    ) -> cluster_pb2.JobStatus:
        """Wait while streaming logs from all tasks using batch log fetching.

        Uses a single batch RPC call per poll interval to fetch logs from all tasks,
        rather than N individual calls. The batch API uses a global since_ms cursor
        for efficient incremental fetching.
        """
        since_ms = 0
        stream_interval = Duration.from_seconds(min(0.2, poll_interval))
        deadline = Deadline.from_seconds(timeout)

        while True:
            status = self._client._cluster_client.get_job_status(self._job_id)

            try:
                result = self._client.stream_task_logs(
                    self._job_id,
                    include_children=include_children,
                    since_ms=since_ms,
                )

                _log_task_results(result)

                if result.last_timestamp_ms > since_ms:
                    since_ms = result.last_timestamp_ms
            except Exception as e:
                logger.warning("Failed to fetch job logs: %s", e)

            if is_job_finished(status.state):
                # Final drain to catch any remaining logs
                try:
                    result = self._client.stream_task_logs(
                        self._job_id,
                        include_children=include_children,
                        since_ms=since_ms,
                    )
                    _log_task_results(result)
                except Exception as e:
                    logger.warning("Failed to fetch final job logs: %s", e)
                return status

            deadline.raise_if_expired(f"Job {self._job_id} did not complete in {timeout}s")
            time.sleep(stream_interval.to_seconds())

    def terminate(self) -> None:
        """Terminate this job."""
        self._client._cluster_client.terminate_job(self._job_id)


# =============================================================================
# Context Management
# =============================================================================


class EndpointRegistry(Protocol):
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
    """Endpoint registry that auto-prefixes names with a namespace."""

    def __init__(
        self,
        cluster: LocalClusterClient | RemoteClusterClient,
        namespace: Namespace,
        job_id: JobName,
    ):
        self._cluster = cluster
        self._namespace = namespace
        self._job_id = job_id

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
        if name.startswith("/") or not self._namespace:
            prefixed_name = name
        else:
            prefixed_name = f"{self._namespace}/{name}"

        return self._cluster.register_endpoint(
            name=prefixed_name,
            address=address,
            job_id=self._job_id,
            metadata=metadata,
        )

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint.

        Args:
            endpoint_id: Endpoint ID to remove
        """
        self._cluster.unregister_endpoint(endpoint_id)


class NamespacedResolver:
    """Resolver that auto-prefixes names with namespace."""

    def __init__(self, cluster: LocalClusterClient | RemoteClusterClient, namespace: Namespace | None = None):
        self._cluster = cluster
        self._namespace = namespace

    def resolve(self, name: str) -> ResolveResult:
        """Resolve actor name to endpoints.

        The name is auto-prefixed with the namespace before lookup.

        Args:
            name: Actor name to resolve (will be prefixed)

        Returns:
            ResolveResult with matching endpoints
        """
        if name.startswith("/"):
            prefixed_name = name
        elif self._namespace:
            prefixed_name = f"{self._namespace}/{name}"
        else:
            prefixed_name = name

        logger.debug("NamespacedResolver resolving: %s", prefixed_name)
        matches = self._cluster.list_endpoints(prefix=prefixed_name)
        logger.debug(
            "NamespacedResolver %s => %s",
            prefixed_name,
            [{"name": ep.name, "id": ep.endpoint_id, "address": ep.address} for ep in matches],
        )

        # Filter to exact matches
        endpoints = [
            ResolvedEndpoint(
                url=ep.address,
                actor_id=ep.endpoint_id,
                metadata=dict(ep.metadata),
            )
            for ep in matches
            if ep.name == prefixed_name
        ]

        return ResolveResult(name=name, endpoints=endpoints)


@dataclass
class LocalClientConfig:
    """Configuration for local job execution.

    Attributes:
        max_workers: Maximum concurrent job threads
    """

    max_workers: int = 4


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

    def __init__(self, cluster: LocalClusterClient | RemoteClusterClient, namespace: Namespace = Namespace("")):
        """Initialize IrisClient with a cluster client.

        Prefer using factory methods (local(), remote()) over direct construction.

        Args:
            cluster: Low-level cluster client (LocalClusterClient or RemoteClusterClient)
        """
        self._cluster_client = cluster
        self._namespace = namespace

    @classmethod
    def local(cls, config: LocalClientConfig | None = None) -> "IrisClient":
        """Create an IrisClient for local execution using real Controller/Worker.

        Args:
            config: Configuration for local execution

        Returns:
            IrisClient wrapping LocalClusterClient
        """
        cfg = config or LocalClientConfig()
        cluster = LocalClusterClient.create(max_workers=cfg.max_workers)
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

    def resolver_for_job(self, job_id: JobName) -> Resolver:
        """Get a resolver for endpoints registered by a specific job.

        Use this when resolving endpoints from outside a job context, such as
        from WorkerPool which runs in client context but needs to resolve
        endpoints registered by its worker jobs.

        Args:
            job_id: The job whose namespace to resolve endpoints in

        Returns:
            Resolver that prefixes lookups with the job's namespace
        """
        namespace = Namespace.from_job_id(job_id)
        return NamespacedResolver(self._cluster_client, namespace=namespace)

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
        ports: list[str] | None = None,
        scheduling_timeout: Duration | None = None,
        constraints: list[Constraint] | None = None,
        coscheduling: CoschedulingConfig | None = None,
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 100,
        timeout: Duration | None = None,
        fail_if_exists: bool = False,
    ) -> Job:
        """Submit a job with automatic job_id hierarchy.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])
            scheduling_timeout: Maximum time to wait for scheduling (None = no timeout)
            constraints: Constraints for filtering workers by attribute
            coscheduling: Configuration for atomic multi-task scheduling
            replicas: Number of tasks to create for gang scheduling (default: 1)
            max_retries_failure: Max retries per task on failure (default: 0)
            max_retries_preemption: Max retries per task on preemption (default: 100)
            timeout: Per-task timeout (None = no timeout)
            fail_if_exists: If True, return ALREADY_EXISTS error even if an existing
                job with the same name is finished. If False (default), finished jobs
                are automatically replaced.

        Returns:
            Job handle for the submitted job

        Raises:
            ValueError: If name contains '/' or replicas < 1
        """
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")

        # Get parent job ID from context
        ctx = get_iris_ctx()
        parent_job_id = ctx.job_id if ctx else None

        # Construct full hierarchical name
        if parent_job_id:
            job_id = parent_job_id.child(name)
        else:
            job_id = JobName.root(name)

        # If running inside a job, inherit env vars, extras, and pip_packages from parent.
        # Child-specified values take precedence over inherited ones.
        if parent_job_id:
            job_info = get_job_info()
            inherited = dict(job_info.env) if job_info else {}
            child_env = {**inherited, **(environment.env_vars or {})} if environment else inherited

            parent_extras = job_info.extras if job_info else []
            parent_pip = job_info.pip_packages if job_info else []

            if environment:
                environment = EnvironmentSpec(
                    pip_packages=environment.pip_packages or parent_pip,
                    env_vars=child_env,
                    extras=environment.extras or parent_extras,
                )
            else:
                environment = EnvironmentSpec(
                    env_vars=child_env,
                    extras=parent_extras,
                    pip_packages=parent_pip,
                )

        # Convert to wire format
        resources_proto = resources.to_proto()
        environment_proto = environment.to_proto() if environment else None
        constraints_proto = [c.to_proto() for c in constraints or []]
        coscheduling_proto = coscheduling.to_proto() if coscheduling else None

        self._cluster_client.submit_job(
            job_id=job_id,
            entrypoint=entrypoint,
            resources=resources_proto,
            environment=environment_proto,
            ports=ports,
            scheduling_timeout=scheduling_timeout,
            constraints=constraints_proto,
            coscheduling=coscheduling_proto,
            replicas=replicas,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            timeout=timeout,
            fail_if_exists=fail_if_exists,
        )

        return Job(self, job_id)

    def status(self, job_id: JobName) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobStatus proto with current state
        """
        return self._cluster_client.get_job_status(job_id)

    def terminate(self, job_id: JobName) -> None:
        """Terminate a running job.

        Args:
            job_id: Job ID to terminate
        """
        self._cluster_client.terminate_job(job_id)

    def list_jobs(
        self,
        *,
        states: list[cluster_pb2.JobState] | None = None,
        prefix: JobName | None = None,
    ) -> list[cluster_pb2.JobStatus]:
        """List jobs with optional filtering.

        Args:
            states: If provided, only return jobs in these states
            prefix: If provided, only return jobs whose JobName starts with this prefix

        Returns:
            List of JobStatus matching the filters
        """
        all_jobs = self._cluster_client.list_jobs()
        result = []
        for job in all_jobs:
            if states is not None and job.state not in states:
                continue
            job_name = JobName.from_wire(job.job_id)
            if prefix is not None and not job_name.to_wire().startswith(prefix.to_wire()):
                continue
            result.append(job)
        return result

    def terminate_prefix(
        self,
        prefix: JobName,
        *,
        exclude_finished: bool = True,
    ) -> list[JobName]:
        """Terminate all jobs matching a prefix.

        Args:
            prefix: Job name prefix to match (e.g., JobName.root("my-experiment"))
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
            job_id = JobName.from_wire(job.job_id)
            self.terminate(job_id)
            terminated.append(job_id)
        return terminated

    def task_status(self, task_name: JobName) -> cluster_pb2.TaskStatus:
        """Get status of a specific task.

        Args:
            task_name: Full task name (/job/.../index)

        Returns:
            TaskStatus proto containing state, worker assignment, and metrics
        """
        return self._cluster_client.get_task_status(task_name)

    def list_tasks(self, job_id: JobName) -> list[cluster_pb2.TaskStatus]:
        """List all tasks for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of TaskStatus protos, one per task
        """
        return self._cluster_client.list_tasks(job_id)

    def fetch_task_logs(
        self,
        target: JobName,
        *,
        include_children: bool = False,
        start: Timestamp | None = None,
        max_lines: int = 0,
        regex: str | None = None,
        attempt_id: int = -1,
    ) -> list[TaskLogEntry]:
        """Fetch logs for a task or job.

        Args:
            target: Task ID or Job ID (detected by trailing numeric)
            include_children: Include logs from child jobs (job ID only)
            start: Only return logs after this timestamp (None = from beginning)
            max_lines: Maximum number of log lines to return (0 = unlimited)
            regex: Regex filter for log content
            attempt_id: Filter to specific attempt (-1 = all attempts)

        Returns:
            List of TaskLogEntry objects, sorted by timestamp
        """
        response = self._cluster_client.fetch_task_logs(
            target,
            include_children=include_children,
            since_ms=start.epoch_ms() if start else 0,
            max_total_lines=max_lines,
            regex=regex,
            attempt_id=attempt_id,
        )

        result: list[TaskLogEntry] = []
        for batch in response.task_logs:
            task_id = JobName.from_wire(batch.task_id)
            worker_id = batch.worker_id or ""
            for proto in batch.logs:
                result.append(
                    TaskLogEntry(
                        timestamp=Timestamp.from_proto(proto.timestamp),
                        worker_id=worker_id,
                        task_id=task_id,
                        source=proto.source,
                        data=proto.data,
                        attempt_id=proto.attempt_id,
                    )
                )

        result.sort(key=lambda x: x.timestamp.epoch_ms())
        return result

    def stream_task_logs(
        self,
        target: JobName,
        *,
        include_children: bool = False,
        since_ms: int = 0,
        max_lines: int = 0,
        regex: str | None = None,
        attempt_id: int = -1,
    ) -> TaskLogsResult:
        """Fetch logs for a task or job with full context.

        Returns structured results including task/worker context and any errors
        encountered while fetching logs. Entries are sorted by timestamp.

        Args:
            target: Task ID or Job ID (detected by trailing numeric)
            include_children: Include logs from child jobs (job ID only)
            since_ms: Only return logs after this timestamp in epoch ms (exclusive)
            max_lines: Maximum number of log lines to return (0 = unlimited)
            regex: Regex filter for log content
            attempt_id: Filter to specific attempt (-1 = all attempts)

        Returns:
            TaskLogsResult with entries, errors, and metadata
        """
        response = self._cluster_client.fetch_task_logs(
            target,
            include_children=include_children,
            since_ms=since_ms,
            max_total_lines=max_lines,
            regex=regex,
            attempt_id=attempt_id,
        )

        entries: list[TaskLogEntry] = []
        errors: list[TaskLogError] = []

        for batch in response.task_logs:
            task_id = JobName.from_wire(batch.task_id)
            worker_id = batch.worker_id or ""

            if batch.error:
                errors.append(TaskLogError(task_id=task_id, worker_id=worker_id, error=batch.error))

            for proto in batch.logs:
                entries.append(
                    TaskLogEntry(
                        timestamp=Timestamp.from_proto(proto.timestamp),
                        worker_id=worker_id or "?",
                        task_id=task_id,
                        source=proto.source,
                        data=proto.data,
                        attempt_id=proto.attempt_id,
                    )
                )

        entries.sort(key=lambda e: e.timestamp.epoch_ms())

        return TaskLogsResult(
            entries=entries,
            errors=errors,
            last_timestamp_ms=response.last_timestamp_ms,
            truncated=response.truncated,
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the client.

        Args:
            wait: If True, wait for pending jobs to complete (local mode only)
        """
        self._cluster_client.shutdown(wait=wait)


@dataclass
class IrisContext:
    """Unified execution context for Iris.

    Available in any iris job via `iris_ctx()`. Contains all
    information about the current execution environment.

    Attributes:
        job_id: Unique identifier for this job (hierarchical: "/root/parent/child")
        attempt_id: Attempt number for this job execution (0-based)
        worker_id: Identifier for the worker executing this job (may be None)
        client: IrisClient for job operations (submit, status, wait, etc.)
        ports: Allocated ports by name (e.g., {"actor": 50001})
    """

    job_id: JobName | None
    attempt_id: int = 0
    worker_id: str | None = None
    client: "IrisClient | None" = None
    ports: dict[str, int] | None = None

    def __post_init__(self):
        if self.ports is None:
            self.ports = {}

    @property
    def registry(self) -> NamespacedEndpointRegistry:
        """Endpoint registry for this job context. Creates on demand.

        Raises:
            RuntimeError: If no client is available
        """
        if self.client is None:
            raise RuntimeError("No client available - ensure controller_address is set")
        if self.job_id is None:
            raise RuntimeError("No job id available - ensure IrisContext is initialized from a job")
        return NamespacedEndpointRegistry(
            self.client._cluster_client,
            self.namespace,
            self.job_id,
        )

    @property
    def namespace(self) -> Namespace:
        """Namespace derived from the root job ID.

        All jobs in a hierarchy share the same namespace, enabling actors
        to be discovered across the job tree.
        """
        if self.job_id is None:
            raise RuntimeError("No job id available - ensure IrisContext is initialized from a job")
        return Namespace.from_job_id(self.job_id)

    @property
    def parent_job_id(self) -> JobName | None:
        """Parent job ID, or None if this is a root job.

        For job_id "/root/parent/child", returns "/root/parent".
        For job_id "/root", returns None.
        """
        if self.job_id is None:
            return None
        return self.job_id.parent

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
        return NamespacedResolver(self.client._cluster_client, self.namespace)

    @staticmethod
    def from_job_info(
        info: JobInfo,
        client: "IrisClient | None" = None,
    ) -> "IrisContext":
        """Create IrisContext from JobInfo.

        Args:
            info: JobInfo from cluster layer
            client: Optional IrisClient instance

        Returns:
            IrisContext with metadata from JobInfo
        """
        return IrisContext(
            job_id=info.job_id,
            attempt_id=info.attempt_id,
            worker_id=info.worker_id,
            client=client,
            ports=dict(info.ports),
        )


# Module-level ContextVar for the current iris context
_iris_context: ContextVar[IrisContext | None] = ContextVar(
    "iris_context",
    default=None,
)


def iris_ctx() -> IrisContext:
    """Get the current IrisContext, raising if not in a job.

    Returns:
        Current IrisContext

    Raises:
        RuntimeError: If not running inside an Iris job
    """
    ctx = get_iris_ctx()
    if ctx is None:
        raise RuntimeError("iris_ctx() called outside an Iris job (no job info available)")
    return ctx


def get_iris_ctx() -> IrisContext | None:
    """Get the current IrisContext, or None if not in a job.

    Checks the ContextVar first. If unset, checks whether we're inside an
    Iris job (via get_job_info) and auto-creates the context if so.

    Returns:
        Current IrisContext or None
    """
    ctx = _iris_context.get()
    if ctx is not None:
        return ctx

    # Get job info from environment
    job_info = get_job_info()
    if job_info is None:
        return None

    else:
        # Set up client if controller address is available
        client = None
        if job_info.controller_address:
            bundle_gcs_path = job_info.bundle_gcs_path

            # Create remote client for context use
            client = IrisClient.remote(
                controller_address=job_info.controller_address,
                bundle_gcs_path=bundle_gcs_path,
            )

        ctx = IrisContext.from_job_info(job_info, client=client)
    _iris_context.set(ctx)
    return ctx


@contextmanager
def iris_ctx_scope(ctx: IrisContext) -> Generator[IrisContext, None, None]:
    """Set the iris context for the duration of this scope.

    Args:
        ctx: Context to set for this scope

    Yields:
        The provided context

    Example:
        ctx = IrisContext(job_id=JobName.from_string("/my-namespace/job-1"), worker_id="worker-1")
        with iris_ctx_scope(ctx):
            my_job_function()
    """
    token = _iris_context.set(ctx)
    try:
        yield ctx
    finally:
        _iris_context.reset(token)
