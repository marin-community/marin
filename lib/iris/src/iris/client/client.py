# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""High-level client with automatic job hierarchy and namespace-based actor discovery.

Example:
    # In job code:
    from iris.client.client import iris_ctx

    ctx = iris_ctx()
    print(f"Running job {ctx.job_id} in namespace {ctx.namespace}")

    # Get allocated port for actor server
    port = ctx.get_port("actor")

    # Submit a sub-job
    sub_job_id = ctx.client.submit(entrypoint, "sub-job", resources)
"""

import logging
import re
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Protocol, TypeVar, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.credentials import ClientCredentials
from rigging.timing import Deadline, Duration, ExponentialBackoff, Timestamp

from iris.actor.resolver import ResolvedEndpoint, Resolver, ResolveResult
from iris.client.context_state import current_context, reset_context, set_context
from iris.client.workload import AttemptStatus, JobStatus, TaskActionResult, TaskDescription, TaskStatus
from iris.client.workload_codec import (
    job_state_from_proto,
    job_status_from_proto,
    task_action_result_from_proto,
    task_description_from_proto,
    task_status_from_proto,
)
from iris.cluster.client import (
    ClusterClient,
    JobInfo,
    RemoteClusterClient,
    get_job_info,
    resolve_job_user,
)
from iris.cluster.constraints import (
    Constraint,
    is_any_region_marker,
    merge_constraints,
)
from iris.cluster.log_keys import build_log_source
from iris.cluster.types import (
    CoschedulingConfig,
    EndpointAccess,
    Entrypoint,
    EnvironmentSpec,
    JobName,
    Namespace,
    ResourceSpec,
    TaskAttempt,
    adjust_tpu_replicas,
)
from iris.resources.state import TERMINAL_TASK_STATES, JobState, TaskState, is_job_finished
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import timestamp_from_proto

logger = logging.getLogger(__name__)


class _ClusterLifecycle(Protocol):
    """Anything IrisClient owns and tears down on shutdown — typically a LocalCluster."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskLogEntry:
    """A log entry with task context.

    Attributes:
        timestamp: When the log line was produced
        task_id: Task that produced this log
        source: Log source - "stdout", "stderr", or "build"
        data: Log line content
        attempt_id: Which attempt produced this log (0-indexed)
        key: Log store key (populated on multi-key queries)
    """

    timestamp: Timestamp
    task_id: JobName
    source: str
    data: str
    attempt_id: int = 0
    key: str = ""


@dataclass(frozen=True, slots=True)
class _LogQuery:
    start: Timestamp | None = None
    max_lines: int = 0
    substring: str = ""
    min_level: str = ""
    tail: bool = False


def _task_id_from_key(key: str, fallback: JobName | None = None) -> JobName:
    """Extract the task JobName from a log entry key (e.g. "/user/job/0:3" -> "/user/job/0")."""
    if not key:
        if fallback is None:
            raise ValueError("Log entry omitted its task key")
        fallback.require_task()
        return fallback
    colon = key.rfind(":")
    if colon >= 0:
        return JobName.from_wire(key[:colon])
    return JobName.from_wire(key)


def _require_job_name(job_id: JobName) -> JobName:
    if job_id.is_task:
        raise ValueError(f"Expected a Job name, got Task {job_id}")
    return job_id


class JobFailedError(Exception):
    """Raised when a job ends in a state other than SUCCEEDED."""

    def __init__(self, job_id: JobName, status: JobStatus):
        self.job_id = job_id
        self.status = status
        state_name = status.state.name
        msg = f"Job {job_id} {state_name}"
        if status.error_message:
            msg += f": {status.error_message}"
        super().__init__(msg)


class JobAlreadyExists(Exception):
    """Raised when a job with the same name is already running."""

    def __init__(self, message: str):
        super().__init__(message)


_Status = TypeVar("_Status")


def _wait_for_status(
    load: Callable[[Deadline], _Status],
    finished: Callable[[_Status], bool],
    *,
    timeout: float,
    poll_interval: float,
    target: str,
) -> _Status:
    deadline = Deadline.from_seconds(timeout)
    backoff = ExponentialBackoff(initial=0.1, maximum=max(0.1, poll_interval))
    while True:
        status = load(deadline)
        if finished(status):
            return status
        deadline.raise_if_expired(f"{target} did not finish in {timeout}s")
        Event().wait(min(backoff.next_interval(), deadline.remaining_seconds()))


class Attempt:
    """Handle for one numbered Attempt of a logical Task."""

    def __init__(self, client: "IrisClient", task_name: JobName, attempt_number: int):
        task_name.require_task()
        if attempt_number < 0:
            raise ValueError("attempt_number must be non-negative")
        self._client = client
        self._task_name = task_name
        self._attempt_number = attempt_number

    @property
    def task_id(self) -> JobName:
        return self._task_name

    @property
    def job_id(self) -> JobName:
        return TaskAttempt(self._task_name).job_id

    @property
    def attempt_number(self) -> int:
        return self._attempt_number

    @property
    def ref(self) -> TaskAttempt:
        return TaskAttempt(self._task_name, self._attempt_number)

    def _status(self, deadline: Deadline | None) -> AttemptStatus:
        """Return this numbered Attempt from the Task's retained history."""
        status = self._client.task_status(self._task_name, deadline=deadline)
        match = next(
            (attempt for attempt in status.attempts if attempt.attempt_number == self._attempt_number),
            None,
        )
        if match is None:
            raise ConnectError(Code.NOT_FOUND, f"Attempt {self.ref.to_wire()} not found")
        return match

    def status(self) -> AttemptStatus:
        """Return this numbered Attempt from the Task's retained history."""
        return self._status(None)

    def logs(
        self,
        *,
        start: Timestamp | None = None,
        max_lines: int = 0,
        substring: str = "",
        min_level: str = "",
        tail: bool = False,
    ) -> list[TaskLogEntry]:
        """Fetch logs for this numbered Attempt."""
        return self._client._fetch_logs(
            self._task_name,
            _LogQuery(
                start=start,
                max_lines=max_lines,
                substring=substring,
                min_level=min_level,
                tail=tail,
            ),
            attempt_id=self._attempt_number,
        )

    def wait(self, timeout: float = 300.0, poll_interval: float = 30.0) -> AttemptStatus:
        """Wait until this Attempt reaches a terminal state."""
        return _wait_for_status(
            self._status,
            lambda status: status.state in TERMINAL_TASK_STATES,
            timeout=timeout,
            poll_interval=poll_interval,
            target=f"Attempt {self.ref.to_wire()}",
        )

    def preempt(self, *, reason: str = "") -> TaskActionResult:
        """Preempt this Attempt if it is still current."""
        return self._client.preempt_tasks((self.ref,), reason=reason)[0]

    def fail(self, *, reason: str = "") -> TaskActionResult:
        """Fail this Attempt without retry if it is still current."""
        return self._client.fail_tasks((self.ref,), reason=reason)[0]


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
        task_name.require_task()
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

    def status(self) -> TaskStatus:
        """Return the current snapshot for this logical Task."""
        return self._client.task_status(self.task_id)

    def describe(self) -> TaskDescription:
        """Return submitted resources and failure diagnostics with the Task snapshot."""
        return self._client.describe_task(self.task_id)

    @property
    def state(self) -> TaskState:
        """Get current task state (shortcut for status().state)."""
        return self.status().state

    def attempts(self) -> tuple[Attempt, ...]:
        """Return handles for the retained Attempt history."""
        return tuple(Attempt(self._client, self._task_name, item.attempt_number) for item in self.status().attempts)

    def attempt(self, attempt_number: int) -> Attempt:
        """Address one numbered Attempt."""
        return Attempt(self._client, self._task_name, attempt_number)

    def current_attempt(self) -> Attempt | None:
        """Return the current Attempt, or None before the first Attempt exists."""
        status = self.status()
        if not any(item.attempt_number == status.current_attempt_number for item in status.attempts):
            return None
        return Attempt(self._client, self._task_name, status.current_attempt_number)

    def logs(
        self,
        *,
        start: Timestamp | None = None,
        max_lines: int = 0,
        substring: str = "",
        min_level: str = "",
        tail: bool = False,
    ) -> list[TaskLogEntry]:
        """Fetch logs across this Task's Attempts."""
        return self._client._fetch_logs(
            self._task_name,
            _LogQuery(
                start=start,
                max_lines=max_lines,
                substring=substring,
                min_level=min_level,
                tail=tail,
            ),
        )

    def wait(self, timeout: float = 300.0, poll_interval: float = 30.0) -> TaskStatus:
        """Wait until this Task reaches a terminal state."""
        return _wait_for_status(
            lambda deadline: self._client.task_status(self.task_id, deadline=deadline),
            lambda status: status.state in TERMINAL_TASK_STATES,
            timeout=timeout,
            poll_interval=poll_interval,
            target=f"Task {self._task_name}",
        )

    def preempt(self, *, reason: str = "") -> TaskActionResult:
        """Preempt the current Attempt under the Task retry policy."""
        return self._client.preempt_tasks((TaskAttempt(self._task_name),), reason=reason)[0]

    def fail(self, *, reason: str = "") -> TaskActionResult:
        """Fail the current Attempt without retry."""
        return self._client.fail_tasks((TaskAttempt(self._task_name),), reason=reason)[0]


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
        _require_job_name(job_id)
        self._client = client
        self._job_id = job_id

    @property
    def job_id(self) -> JobName:
        """Logical Job name."""
        return self._job_id

    def __str__(self) -> str:
        return str(self._job_id)

    def __repr__(self) -> str:
        return f"Job({self._job_id!r})"

    def status(self) -> JobStatus:
        """Return the current snapshot for this logical Job."""
        return self._client.job_status(self._job_id)

    def state_only(self) -> JobState:
        """Lightweight state query that avoids loading tasks/attempts/workers."""
        return self._client.job_state(self._job_id)

    @property
    def state(self) -> JobState:
        """Get current job state via the lightweight state-only RPC."""
        return self.state_only()

    def tasks(self) -> list[Task]:
        """Get all tasks for this job.

        Returns:
            List of Task handles, one per task in the job
        """
        return [Task(self._client, status.task_id) for status in self._client.list_tasks(self._job_id)]

    def task(self, task_index: int) -> Task:
        """Address one Task by its zero-based index."""
        if task_index < 0:
            raise ValueError("task_index must be non-negative")
        return Task(self._client, self._job_id.task(task_index))

    def logs(
        self,
        *,
        start: Timestamp | None = None,
        max_lines: int = 0,
        substring: str = "",
        min_level: str = "",
        tail: bool = False,
    ) -> list[TaskLogEntry]:
        """Fetch globally timestamp-ordered logs across this job's tasks.

        Args:
            start: Only return entries after this timestamp.
            max_lines: Global maximum number of lines to return. Zero uses the server default.
            substring: Only return entries containing this text.
            min_level: Minimum log level to return.
            tail: Return the most recent lines instead of the earliest lines.
        """
        return self._client._fetch_logs(
            self._job_id,
            _LogQuery(
                start=start,
                max_lines=max_lines,
                substring=substring,
                min_level=min_level,
                tail=tail,
            ),
        )

    def wait(
        self,
        timeout: float = 300.0,
        poll_interval: float = 30.0,
        *,
        raise_on_failure: bool = True,
        stream_logs: bool = False,
        since_ms: int = 0,
        min_level: str = "",
        substring: str = "",
    ) -> JobStatus:
        """Wait for job to complete.

        Args:
            timeout: Maximum wait time in seconds
            poll_interval: Upper bound on the state-poll backoff. The loop
                starts at 100ms and grows exponentially until it reaches this
                cap (default 30s), so long-running jobs cost ~1 state RPC per
                ``poll_interval``.
            raise_on_failure: If True, raise JobFailedError on any non-SUCCESS terminal state
            stream_logs: If True, stream logs from all tasks interleaved
            since_ms: Only show logs after this epoch millisecond timestamp
            min_level: Minimum log level filter (DEBUG/INFO/WARNING/ERROR/CRITICAL)
            substring: Only stream log lines containing this text

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: Job didn't complete in time
            JobFailedError: Job ended in non-SUCCESS state and raise_on_failure=True
        """
        if not stream_logs:
            response = self._client._cluster_client.wait_for_job(self._job_id, timeout, poll_interval)
        else:
            response = self._client._cluster_client.wait_for_job_with_streaming(
                self._job_id,
                timeout=timeout,
                poll_interval=poll_interval,
                since_ms=since_ms,
                min_level=min_level,
                substring=substring,
            )
        status = job_status_from_proto(response)

        if raise_on_failure and status.state is not JobState.SUCCEEDED:
            raise JobFailedError(self._job_id, status)

        return status

    def cancel(self) -> None:
        """Cancel this Job and its descendants."""
        self._client.cancel_job(self._job_id)


# =============================================================================
# Context Management
# =============================================================================


class EndpointRegistry(Protocol):
    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> str:
        """Register an endpoint for actor discovery.

        Args:
            name: Actor name for discovery
            address: Address where actor is listening (host:port)
            metadata: Optional metadata for the endpoint
            access: Proxy access mode — PRIVATE (default), PUBLIC, or BEARER.

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

    def registered(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> AbstractContextManager[str]:
        """Own one renewable endpoint registration for a context lifetime."""
        ...


class NamespacedEndpointRegistry:
    """Endpoint registry that auto-prefixes names with a namespace."""

    def __init__(
        self,
        cluster: ClusterClient,
        namespace: Namespace,
        task_attempt: TaskAttempt,
    ):
        self._cluster = cluster
        self._namespace = namespace
        self._task_attempt = task_attempt

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> str:
        """Register an endpoint, auto-prefixing with namespace.

        Args:
            name: Actor name for discovery (will be prefixed)
            address: Address where actor is listening (host:port)
            metadata: Optional metadata
            access: Proxy access mode — PRIVATE (default), PUBLIC, or BEARER.

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
            task_attempt=self._task_attempt,
            metadata=metadata,
            access=access,
        )

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint.

        Args:
            endpoint_id: Endpoint ID to remove
        """
        self._cluster.unregister_endpoint(endpoint_id)

    @contextmanager
    def registered(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> Generator[str, None, None]:
        """Register and renew an endpoint, then remove it promptly on clean exit."""
        endpoint_id = self.register(name, address, metadata, access)
        try:
            yield endpoint_id
        finally:
            try:
                self.unregister(endpoint_id)
            except Exception:
                logger.warning("Failed to unregister endpoint id=%s", endpoint_id, exc_info=True)


class NamespacedResolver:
    """Resolver that auto-prefixes names with namespace."""

    def __init__(self, cluster: ClusterClient, namespace: Namespace | None = None):
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
        matches = self._cluster.list_endpoint_instances(prefixed_name)
        logger.debug(
            "NamespacedResolver %s => %s",
            prefixed_name,
            [{"name": ep.name, "id": ep.endpoint_id, "address": ep.address} for ep in matches],
        )

        endpoints = [
            ResolvedEndpoint(
                url=ep.address,
                actor_id=ep.endpoint_id,
                metadata=dict(ep.metadata),
            )
            for ep in matches
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
        from iris.client.local_client import make_local_client
        with make_local_client() as client:
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

    def __init__(
        self,
        cluster: ClusterClient,
        controller: _ClusterLifecycle | None = None,
    ):
        """Initialize IrisClient with a cluster client.

        For local execution, prefer ``iris.client.local_client.make_local_client``
        over direct construction; for RPC use ``IrisClient.remote(...)``.

        Args:
            cluster: Low-level cluster client (RemoteClusterClient)
            controller: Optional cluster object whose lifecycle this client owns.
                ``shutdown()`` will call ``controller.close()``.
        """
        self._cluster_client = cluster
        self._controller = controller

    @classmethod
    def remote(
        cls,
        controller_address: str,
        *,
        workspace: Path | None = None,
        bundle_id: str | None = None,
        timeout_ms: int = 30000,
        credentials: ClientCredentials | None = None,
        extra_bundle_includes: Sequence[str] = (),
        bundle_exclude: re.Pattern[str] | None = None,
    ) -> "IrisClient":
        """Create an IrisClient for an external client (CLI, laptop, notebook).

        Finelog logs/stats are routed through the controller, the only ingress
        an external client can reach. In-cluster callers should use
        :meth:`in_cluster` instead.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            workspace: Path to workspace directory containing pyproject.toml.
                If provided, this directory will be bundled and sent to workers.
                Required for external job submission.
            bundle_id: Workspace bundle identifier for sub-job inheritance.
                When set, sub-jobs use this bundle ID instead of creating new bundles.
            timeout_ms: RPC timeout in milliseconds
            credentials: Auth material for outgoing RPCs — the Iris JWT and, for
                an IAP-fronted cluster, the IAP OIDC ID token. None sends neither
                (a loopback-trusted tunnel).
            extra_bundle_includes: Glob patterns (relative to ``workspace``) for
                gitignored files the caller needs in the task bundle — e.g. a package's
                built frontend ``dist``. Bundled in addition to the git-tracked files.
            bundle_exclude: Regex matched against each candidate bundle path
                (POSIX, relative to ``workspace``); matching paths are dropped from
                the bundle. Trims otherwise-tracked files that a job does not need,
                such as ``docs/`` against the bundle size cap.

        Returns:
            IrisClient wrapping RemoteClusterClient
        """
        return cls._make(
            controller_address,
            workspace=workspace,
            bundle_id=bundle_id,
            timeout_ms=timeout_ms,
            credentials=credentials,
            use_controller_proxy=True,
            extra_bundle_includes=extra_bundle_includes,
            bundle_exclude=bundle_exclude,
        )

    @classmethod
    def in_cluster(
        cls,
        controller_address: str,
        *,
        workspace: Path | None = None,
        bundle_id: str | None = None,
        timeout_ms: int = 30000,
        credentials: ClientCredentials | None = None,
    ) -> "IrisClient":
        """Create an IrisClient for code running inside the cluster (in-task).

        Same as :meth:`remote`, except finelog logs/stats are written straight
        to the resolved finelog server instead of through the controller's
        endpoint proxy — so high-frequency task-status pushes don't compete for
        the controller's HTTP proxy. Only valid where the finelog server's
        internal address is reachable (i.e. inside the cluster).
        """
        return cls._make(
            controller_address,
            workspace=workspace,
            bundle_id=bundle_id,
            timeout_ms=timeout_ms,
            credentials=credentials,
            use_controller_proxy=False,
        )

    @classmethod
    def _make(
        cls,
        controller_address: str,
        *,
        workspace: Path | None,
        bundle_id: str | None,
        timeout_ms: int,
        use_controller_proxy: bool,
        credentials: ClientCredentials | None = None,
        extra_bundle_includes: Sequence[str] = (),
        bundle_exclude: re.Pattern[str] | None = None,
    ) -> "IrisClient":
        interceptors = credentials.interceptors() if credentials is not None else []

        cluster = RemoteClusterClient(
            controller_address=controller_address,
            bundle_id=bundle_id,
            workspace=workspace,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
            use_controller_proxy=use_controller_proxy,
            extra_bundle_includes=extra_bundle_includes,
            bundle_exclude=bundle_exclude,
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

    def job(self, job_id: JobName) -> Job:
        """Address an existing logical Job."""
        _require_job_name(job_id)
        return Job(self, job_id)

    def task(self, task_id: JobName) -> Task:
        """Address an existing logical Task."""
        task_id.require_task()
        return Task(self, task_id)

    def attempt(self, ref: TaskAttempt) -> Attempt:
        """Address one numbered Attempt."""
        return Attempt(self, ref.task_id, ref.require_attempt())

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
        max_retries_preemption: int = 1000,
        max_task_failures: int = 0,
        timeout: Duration | None = None,
        user: str | None = None,
        preemption_policy: job_pb2.JobPreemptionPolicy = job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
        existing_job_policy: job_pb2.ExistingJobPolicy = job_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
        task_image: str | None = None,
        priority_band: job_pb2.PriorityBand = job_pb2.PRIORITY_BAND_INHERIT,
        container_profile: job_pb2.ContainerProfile = job_pb2.CONTAINER_PROFILE_UNSPECIFIED,
        submit_argv: list[str] | None = None,
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
            replicas: Number of tasks to create for gang scheduling (default: 1).
                Multi-process GPU execution within a task is composed into the command
                (``python -m iris.hooks.multigpu_main --nproc N -- <cmd>``), not a submit arg.
            max_retries_failure: Max retries per task on failure (default: 0)
            max_retries_preemption: Max retries per task on preemption (default: 100)
            max_task_failures: Cumulative failed task attempts the job tolerates before
                it fails (default: 0 = fail on the first failure). Counts across retries,
                so set this to allow a job to ride out a few inconsistent failures.
            timeout: Per-task timeout (None = no timeout)
            user: Optional explicit user override for top-level jobs
            task_image: Optional override for the task container image. When None,
                the worker uses its cluster-configured default_task_image. Used for
                jobs that need a custom runtime (e.g. an image with runsc/skopeo
                for sandboxing untrusted child workloads).
            container_profile: Container security profile. UNSPECIFIED resolves to
                DEFAULT. Elevated profiles (DOCKER_ACCESS, PRIVILEGED) require the
                admin role at submission when auth is enabled.

        Returns:
            Job handle for the submitted job

        Raises:
            ValueError: If the name is invalid or replicas < 1.
            JobAlreadyExists: If a job with the same name already exists
        """
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")
        replicas = adjust_tpu_replicas(resources.device, replicas)

        # iris is a dumb scheduler: it runs the entrypoint verbatim. Multi-process GPU
        # execution and profiling are composed into the command by the caller
        # (e.g. `python -m iris.hooks.multigpu_main --nproc N -- <cmd>`).

        # Get parent job ID from context
        ctx = get_iris_ctx()
        parent_job_id = ctx.job_id if ctx else None
        if parent_job_id is not None and parent_job_id.child(name).is_task:
            raise ValueError(f"Nested Job name cannot be an integer: {name!r}")

        # Construct full hierarchical name
        if parent_job_id:
            job_id = parent_job_id.child(name)
        else:
            job_id = JobName.root(resolve_job_user(user), name)

        # If running inside a job, inherit env vars and the parent's resolved setup
        # from the parent. A child that specifies its own setup (explicit
        # setup_scripts, or builder inputs to rebuild the default) takes control of
        # its environment; one that specifies only env vars (or nothing) reuses the
        # parent's setup so it lands in the same environment.
        if parent_job_id:
            job_info = get_job_info()
            inherited = dict(job_info.env) if job_info else {}
            child_env = {**inherited, **(environment.env_vars or {})} if environment else inherited

            parent_setup_scripts = job_info.setup_scripts if job_info else None

            if environment:
                child_owns_setup = (
                    environment.setup_scripts is not None
                    or environment.extras
                    or environment.pip_packages
                    or environment.sync_packages
                )
                environment = EnvironmentSpec(
                    pip_packages=environment.pip_packages,
                    env_vars=child_env,
                    extras=environment.extras,
                    setup_scripts=environment.setup_scripts if child_owns_setup else parent_setup_scripts,
                    sync_packages=environment.sync_packages,
                )
            else:
                environment = EnvironmentSpec(env_vars=child_env, setup_scripts=parent_setup_scripts)

            parent_constraints = list(job_info.constraints) if job_info else []
            if constraints is None:
                constraints = parent_constraints
            elif len(constraints) == 0:
                constraints = []
            else:
                constraints = merge_constraints(parent_constraints, constraints)

        # The ANY-region marker clears inherited region constraints during merging.
        # Drop it before the wire so it does not exclude workers without region metadata.
        if constraints:
            constraints = [c for c in constraints if not is_any_region_marker(c)]

        # Convert to wire format
        resources_proto = resources.to_proto()
        environment_proto = environment.to_proto() if environment else None
        constraints_proto = [c.to_proto() for c in constraints or []]
        coscheduling_proto = coscheduling.to_proto() if coscheduling else None

        try:
            canonical_id = self._cluster_client.submit_job(
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
                max_task_failures=max_task_failures,
                timeout=timeout,
                preemption_policy=preemption_policy,
                existing_job_policy=existing_job_policy,
                task_image=task_image,
                priority_band=priority_band,
                container_profile=container_profile,
                submit_argv=submit_argv,
            )
        except ConnectError as e:
            if e.code == Code.ALREADY_EXISTS:
                raise JobAlreadyExists(str(e)) from e
            raise

        return Job(self, canonical_id)

    def job_status(self, job_id: JobName) -> JobStatus:
        """Return the current snapshot for a logical Job name."""
        _require_job_name(job_id)
        return job_status_from_proto(self._cluster_client.get_job_status(job_id))

    def job_state(self, job_id: JobName) -> JobState:
        """Lightweight state query that avoids loading tasks/attempts/workers.

        Prefer this over ``job_status(job_id).state`` for polling loops.
        """
        _require_job_name(job_id)
        states = self._cluster_client.get_job_states([job_id])
        wire_id = job_id.to_wire()
        if wire_id not in states:
            raise ConnectError(Code.NOT_FOUND, f"Job {wire_id} not found")
        return job_state_from_proto(states[wire_id])

    def cancel_job(self, job_id: JobName) -> None:
        """Cancel a running Job and its descendants.

        Args:
            job_id: Job ID to cancel
        """
        _require_job_name(job_id)
        self._cluster_client.terminate_job(job_id)

    def list_jobs(
        self,
        *,
        state: JobState | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[JobStatus]:
        """List jobs with optional filtering.

        Filters are pushed down to the server via ``JobQuery``: ``state``
        becomes ``state_filter`` and ``prefix`` becomes ``job_id_prefix``, an
        anchored prefix match against the wire-form job_id (e.g.
        ``"/alice/exp-"``). The prefix is passed through verbatim; callers do
        not need to provide a parseable ``JobName``.

        Args:
            state: If provided, only return jobs in this state.
            prefix: If provided, only return jobs whose ``job_id`` (wire form,
                e.g. ``"/alice/foo"``) starts with this string.
            limit: If provided, return at most this many jobs (the most recent,
                since the server sorts by submission date descending). ``None``
                walks every matching job, which requires a filter narrow enough
                to stay under the server's deep-offset cap.

        Returns:
            List of JobStatus matching the filters.
        """
        query = controller_pb2.Controller.JobQuery()
        if state is not None:
            query.state_filter = state.value
        if prefix:
            query.job_id_prefix = prefix

        return [job_status_from_proto(job) for job in self._cluster_client.list_jobs(query=query, limit=limit)]

    def list_workers(
        self,
        query: controller_pb2.Controller.WorkerQuery | None = None,
    ) -> list[controller_pb2.Controller.WorkerHealthStatus]:
        """List workers registered with the controller."""
        return list(self._cluster_client.list_workers(query=query))

    def active_job_names_for_prefix(self, prefix: str) -> list[JobName]:
        """Return nonterminal jobs whose wire IDs start with ``prefix`` verbatim."""
        return [job.job_id for job in self.list_jobs(prefix=prefix) if not is_job_finished(job.state)]

    def cancel_jobs_with_prefix(self, prefix: str) -> list[JobName]:
        """Cancel all active Jobs matching a prefix.

        Args:
            prefix: Wire-form job ID prefix to match (e.g., ``"/alice/my-experiment-"``).

        Returns:
            List of job IDs that were terminated
        """
        job_ids = self.active_job_names_for_prefix(prefix)
        for job_id in job_ids:
            self.cancel_job(job_id)
        return job_ids

    def task_status(self, task_name: JobName, *, deadline: Deadline | None = None) -> TaskStatus:
        """Return the current snapshot for a logical Task name."""
        task_name.require_task()
        return task_status_from_proto(self._cluster_client.get_task_status(task_name, deadline=deadline))

    def describe_task(self, task_name: JobName) -> TaskDescription:
        """Return a Task snapshot with submitted resources and failure diagnostics."""
        task_name.require_task()
        return task_description_from_proto(self._cluster_client.get_task_description(task_name))

    def report_task_status_text(
        self,
        task_id: JobName,
        attempt_id: int,
        detail_md: str,
        summary_md: str,
    ) -> None:
        """Push markdown status text for the running task to finelog (fire-and-forget)."""
        self._cluster_client.report_task_status_text(task_id, attempt_id, detail_md, summary_md)

    def resolve_endpoint(self, url: str) -> str:
        """Resolve a logical endpoint URL to a concrete HTTP address via the controller registry."""
        return self._cluster_client.resolve_endpoint(url)

    def list_endpoints(self, prefix: str) -> list[controller_pb2.Controller.Endpoint]:
        """List registered endpoints matching a name prefix."""
        return self._cluster_client.list_endpoints(prefix)

    def list_endpoint_instances(self, name: str) -> list[controller_pb2.Controller.Endpoint]:
        """List registered instances with the exact endpoint name."""
        return self._cluster_client.list_endpoint_instances(name)

    def mint_endpoint_token(
        self,
        endpoint_name: str,
        *,
        ttl: Duration | None = None,
    ) -> controller_pb2.Controller.MintEndpointTokenResponse:
        """Mint a scoped token for a link-accessible endpoint."""
        return self._cluster_client.mint_endpoint_token(endpoint_name, ttl=ttl)

    def list_tasks(self, job_id: JobName) -> list[TaskStatus]:
        """Return current Task snapshots for a logical Job name."""
        _require_job_name(job_id)
        return [task_status_from_proto(task) for task in self._cluster_client.list_tasks(job_id)]

    def _change_tasks(
        self,
        targets: Sequence[TaskAttempt],
        *,
        desired_state: job_pb2.TaskState,
        reason: str,
    ) -> tuple[TaskActionResult, ...]:
        for target in targets:
            target.task_id.require_task()
            if target.attempt_id is not None and target.attempt_id < 0:
                raise ValueError("attempt number must be non-negative")
        wire_targets = [target.to_wire() for target in targets]
        results = self._cluster_client.kick_tasks(wire_targets, desired_state, reason)
        return tuple(task_action_result_from_proto(result) for result in results)

    def preempt_tasks(
        self,
        targets: Sequence[TaskAttempt],
        *,
        reason: str = "",
    ) -> tuple[TaskActionResult, ...]:
        """Preempt current or numbered Attempts under each Task's retry policy."""
        return self._change_tasks(targets, desired_state=job_pb2.TASK_STATE_PREEMPTED, reason=reason)

    def fail_tasks(
        self,
        targets: Sequence[TaskAttempt],
        *,
        reason: str = "",
    ) -> tuple[TaskActionResult, ...]:
        """Fail current or numbered Attempts without retry."""
        return self._change_tasks(targets, desired_state=job_pb2.TASK_STATE_FAILED, reason=reason)

    def _fetch_logs(
        self,
        target: JobName,
        query: _LogQuery,
        *,
        attempt_id: int = -1,
    ) -> list[TaskLogEntry]:
        """Fetch logs for a task or job.

        Builds a literal source + match scope from the target:
        - Task + all attempts:     prefix /user/job/0:
        - Task + specific attempt: exact  /user/job/0:<attempt_id>
        - Job (all tasks):         prefix /user/job/

        Args:
            target: Task ID or Job ID
            query: Log filters and result limits.
            attempt_id: Filter to specific attempt (-1 = all attempts)

        Returns:
            List of TaskLogEntry objects, sorted by timestamp
        """
        source, match_scope = build_log_source(target, attempt_id)
        response = self._cluster_client.fetch_logs(
            source,
            match_scope=match_scope,
            since_ms=query.start.epoch_ms() if query.start else 0,
            max_lines=query.max_lines,
            substring=query.substring,
            min_level=query.min_level,
            tail=query.tail,
        )

        result = [
            TaskLogEntry(
                timestamp=timestamp_from_proto(e.timestamp),
                task_id=_task_id_from_key(e.key, target if attempt_id >= 0 else None),
                source=e.source,
                data=e.data,
                attempt_id=e.attempt_id,
                key=e.key,
            )
            for e in response.entries
        ]
        result.sort(key=lambda x: x.timestamp.epoch_ms())
        return result

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the client and, in local mode, the controller.

        Args:
            wait: If True, wait for pending jobs to complete (local mode only)
        """
        self._cluster_client.shutdown(wait=wait)
        if self._controller is not None:
            self._controller.close()


@dataclass
class IrisContext:
    """Unified execution context for Iris.

    Available in any iris job via `iris_ctx()`. Contains all
    information about the current execution environment.

    Attributes:
        job_id: Unique identifier for this job (hierarchical: "/root/parent/child")
        task_attempt: Structured task identity (task_id + attempt_id). Used for endpoint
            registration so the controller can associate endpoints with the
            specific task and clean them up on retry.
        worker_id: Identifier for the worker executing this job (may be None)
        client: IrisClient for job operations (submit, status, wait, etc.)
        ports: Allocated ports by name (e.g., {"actor": 50001})
    """

    job_id: JobName | None
    task_attempt: TaskAttempt | None = None
    worker_id: str | None = None
    client: "IrisClient | None" = None
    ports: dict[str, int] | None = None

    def __post_init__(self):
        if self.ports is None:
            self.ports = {}

    @property
    def registry(self) -> NamespacedEndpointRegistry:
        """Endpoint registry for this job context. Creates on demand.

        Passes the task_attempt so the controller can associate endpoints with
        the specific task for retry cleanup.

        Raises:
            RuntimeError: If no client or task_attempt is available
        """
        if self.client is None:
            raise RuntimeError("No client available - ensure controller_address is set")
        if self.task_attempt is None:
            raise RuntimeError("No task_attempt available - ensure IrisContext is initialized from a task")
        return NamespacedEndpointRegistry(
            self.client._cluster_client,
            self.namespace,
            self.task_attempt,
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
            task_attempt=info.task_attempt,
            worker_id=info.worker_id,
            client=client,
            ports=dict(info.ports),
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
    ctx = cast(IrisContext | None, current_context())
    if ctx is not None:
        return ctx

    # Get job info from environment
    job_info = get_job_info()
    if job_info is None:
        return None

    # Set up client if controller address is available
    client = None
    if job_info.controller_address:
        bundle_id = job_info.bundle_id
        # In-task code runs inside the cluster and can reach the finelog server
        # directly, so task-status pushes bypass the controller's endpoint proxy.
        client = IrisClient.in_cluster(
            controller_address=job_info.controller_address,
            bundle_id=bundle_id,
        )

    ctx = IrisContext.from_job_info(job_info, client=client)
    set_context(ctx)
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
    token = set_context(ctx)
    try:
        yield ctx
    finally:
        reset_context(token)
