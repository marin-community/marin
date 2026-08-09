# Copyright The Marin Authors
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
import uuid
from collections.abc import Generator, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.credentials import ClientCredentials
from rigging.timing import Deadline, Duration, ExponentialBackoff, Timestamp

from iris.actor.resolver import ResolvedEndpoint, Resolver, ResolveResult
from iris.client.job_info import JobInfo, get_job_info, resolve_job_user
from iris.client.protocol import ClusterClient
from iris.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import (
    Constraint,
    WellKnownAttribute,
    is_any_region_marker,
    merge_constraints,
    region_constraint,
)
from iris.cluster.types import (
    EndpointAccess,
    JobName,
    Namespace,
    TaskAttempt,
)
from iris.resources.action import ActionReceipt
from iris.resources.activity import ActivityEntry, ActivityQuery
from iris.resources.attempt import AttemptDetail
from iris.resources.endpoint import (
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
    ExecResult,
    ProfileConfiguration,
    ProfileResult,
)
from iris.resources.execution import (
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    adjust_tpu_replicas,
    build_runtime_entrypoint,
    with_accelerator_cpu_default,
    with_slice_topology_environment,
)
from iris.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeLocator,
    ResourceKey,
    SliceLocator,
    TaskIdentity,
)
from iris.resources.job import (
    ContainerProfile,
    CoschedulingConfig,
    ExistingJobPolicy,
    JobDetail,
    JobPreemptionPolicy,
    JobQuery,
    JobSpec,
    JobSummary,
    PriorityBand,
)
from iris.resources.log import LogEntry, LogLevel, LogPage, LogQuery
from iris.resources.node import NodeDetail, NodeQuery, NodeSummary
from iris.resources.slice import SliceDetail, SliceQuery, SliceSummary
from iris.resources.source import Page
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.version import client_revision_date

logger = logging.getLogger(__name__)

_RESOURCE_PAGE_SIZE = 500


class _ClusterLifecycle(Protocol):
    """Anything IrisClient owns and tears down on shutdown — typically a LocalCluster."""

    def close(self) -> None: ...


class JobFailedError(Exception):
    """Raised when a job ends in a non-SUCCESS terminal state."""

    def __init__(self, job_id: JobName, status: JobSummary):
        self.job_id = job_id
        self.status = status
        state_name = f"JOB_STATE_{status.state.name}"
        msg = f"Job {job_id} {state_name}"
        if status.error_message:
            msg += f": {status.error_message}"
        super().__init__(msg)


class JobAlreadyExists(Exception):
    """Raised when a job with the same name is already running."""

    def __init__(self, message: str):
        super().__init__(message)


class Task:
    """Handle for a specific task within a job.

    Provides convenient methods for task-level operations like status
    checking and log retrieval.

    Example:
        job = client.submit(entrypoint, "my-job", resources)
        job.wait()
        for task in job.tasks():
            print(f"Task {task.task_index}: {task.state}")
            for entry in task.logs().entries:
                print(entry.data)
    """

    def __init__(self, client: "IrisClient", identity: TaskIdentity):
        self._client = client
        self._identity = identity

    @property
    def identity(self) -> TaskIdentity:
        return self._identity

    @property
    def task_index(self) -> int:
        """0-indexed task number within the job."""
        return JobName.from_wire(self._identity.key.resource_id).require_task()[1]

    @property
    def task_id(self) -> JobName:
        """Full task identifier (/job/.../index)."""
        return JobName.from_wire(self._identity.key.resource_id)

    @property
    def job_id(self) -> JobName:
        """Parent job identifier."""
        return self.task_id.parent or self.task_id

    def describe(self) -> TaskDetail:
        detail = self._client.describe_task(self._identity.key)
        if detail.summary.identity != self._identity:
            raise RuntimeError(f"Task {self.task_id} was replaced")
        return detail

    def status(self) -> TaskSummary:
        """Return the current typed Task summary."""
        return self.describe().summary

    @property
    def state(self) -> TaskState:
        """Get current task state (shortcut for status().state)."""
        return self.status().state

    def logs(self, *, start: Timestamp | None = None, max_lines: int = 1_000) -> LogPage:
        """Fetch exact-incarnation logs for this Task."""
        return self._client.fetch_task_logs(
            self._identity,
            LogQuery(after=start, max_lines=max_lines),
        )

    def retry(self, *, idempotency_key: str | None = None) -> ActionReceipt:
        """Request a retry of the current exact Attempt."""
        current = self.status().current_attempt
        if current is None:
            raise RuntimeError(f"Task {self.task_id} has no current Attempt")
        return self._client.retry_task(
            self._identity,
            expected_attempt_uid=current.attempt_uid,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
        )


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
            for entry in task.logs().entries:
                print(entry.data)
    """

    def __init__(self, client: "IrisClient", identity: JobIdentity):
        self._client = client
        self._identity = identity

    @property
    def identity(self) -> JobIdentity:
        return self._identity

    @property
    def job_id(self) -> JobName:
        """Unique job identifier."""
        return JobName.from_wire(self._identity.key.resource_id)

    def __str__(self) -> str:
        return str(self.job_id)

    def __repr__(self) -> str:
        return f"Job({self._identity!r})"

    def describe(self) -> JobDetail:
        detail = self._client.describe_job(self._identity.key)
        if detail.summary.identity != self._identity:
            raise RuntimeError(f"Job {self.job_id} was replaced")
        return detail

    def status(self) -> JobSummary:
        """Return the current typed Job summary."""
        return self.describe().summary

    def state_only(self) -> JobState:
        """Return the current state of this exact Job incarnation."""
        return self.status().state

    @property
    def state(self) -> JobState:
        """Return the current state of this exact Job incarnation."""
        return self.state_only()

    def tasks(self) -> list[Task]:
        """Get all tasks for this job.

        Returns:
            List of Task handles, one per task in the job
        """
        self.describe()
        tasks: list[Task] = []
        query = TaskQuery(job=self._identity.key)
        while True:
            page = self._client.list_tasks(query)
            tasks.extend(Task(self._client, summary.identity) for summary in page.items)
            if page.next_page_token is None:
                return tasks
            query = TaskQuery(job=self._identity.key, page_token=page.next_page_token)

    def logs(self, *, max_lines: int = 1_000, tail: bool = False) -> LogPage:
        """Fetch exact-incarnation logs across this Job's Tasks."""
        return self._client.fetch_job_logs(
            self._identity,
            LogQuery(max_lines=max_lines, tail=tail),
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
    ) -> JobSummary:
        """Wait for this exact Job incarnation to finish."""
        deadline = Deadline.from_seconds(timeout)
        backoff = ExponentialBackoff(initial=0.1, maximum=max(0.1, poll_interval))
        cursor = 0
        minimum_level = LogLevel[min_level.upper()] if min_level else LogLevel.UNKNOWN
        while True:
            status = self.status()
            if stream_logs:
                page = self._client.fetch_job_logs(
                    self._identity,
                    LogQuery(
                        after=Timestamp.from_ms(since_ms) if since_ms else None,
                        cursor=cursor,
                        minimum_level=minimum_level,
                    ),
                )
                for entry in page.entries:
                    logger.info("task=%s | %s", entry.key, entry.data)
                cursor = max(cursor, page.next_cursor)
            if status.state in {
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.KILLED,
                JobState.WORKER_FAILED,
                JobState.UNSCHEDULABLE,
            }:
                break
            deadline.raise_if_expired(f"Job {self.job_id} did not complete in {timeout}s")
            time.sleep(min(backoff.next_interval(), deadline.remaining_seconds()))

        if raise_on_failure and status.state is not JobState.SUCCEEDED:
            raise JobFailedError(self.job_id, status)
        return status

    def cancel(self, *, idempotency_key: str | None = None) -> ActionReceipt:
        """Cancel this exact Job incarnation."""
        return self._client.cancel_job(
            self._identity,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
        )


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
        matches = self._cluster.resolve_endpoints(prefixed_name)
        logger.debug(
            "NamespacedResolver %s => %s",
            prefixed_name,
            [
                {
                    "name": endpoint.summary.name,
                    "id": endpoint.summary.endpoint_id,
                    "address": endpoint.address,
                }
                for endpoint in matches
            ],
        )

        endpoints = [
            ResolvedEndpoint(
                url=endpoint.address,
                actor_id=endpoint.summary.endpoint_id,
                metadata=dict(endpoint.metadata),
            )
            for endpoint in matches
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
            for entry in task.logs().entries:
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
        max_retries_preemption: int = 1000,
        max_task_failures: int = 0,
        timeout: Duration | None = None,
        user: str | None = None,
        preemption_policy: JobPreemptionPolicy = JobPreemptionPolicy.UNSPECIFIED,
        existing_job_policy: ExistingJobPolicy = ExistingJobPolicy.UNSPECIFIED,
        task_image: str | None = None,
        priority_band: PriorityBand = PriorityBand.INHERIT,
        container_profile: ContainerProfile = ContainerProfile.UNSPECIFIED,
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
            ValueError: If name contains '/' or replicas < 1
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

            # Always inherit the parent's region unless the child already has
            # an explicit region constraint.  This applies even when the caller
            # passes constraints=[] to clear other inherited constraints —
            # region pinning keeps a child co-located with the worker that
            # launched it (where its in-region data and resources live).
            #
            # An explicit region constraint (any op carrying the region key) opts out:
            # a PINNED region (region_constraint) or the ANY marker (any_region_constraint,
            # a region-EXISTS constraint meaning "run anywhere; don't inherit") both satisfy
            # this guard, so neither gets the parent's region appended.
            if job_info and job_info.worker_region and not any(c.key == WellKnownAttribute.REGION for c in constraints):
                inherited_region = region_constraint([job_info.worker_region])
                constraints = [*constraints, inherited_region]

        # The ANY-region marker's only job is the inheritance opt-out above (and clearing
        # any inherited pin via merge_constraints). Once that decision is made it carries no
        # requirement, so drop it before the wire: as a hard region-EXISTS constraint it
        # would otherwise exclude every worker/scaling group that advertises no region
        # attribute. Stripping here keeps the controller's matching paths unaware of it.
        if constraints:
            constraints = [c for c in constraints if not is_any_region_marker(c)]

        # Lower the ergonomic submission inputs into the resource contract.
        resources = with_accelerator_cpu_default(resources)
        resolved_environment = (environment or EnvironmentSpec()).resolve()
        resolved_environment = with_slice_topology_environment(resolved_environment, resources, replicas)
        runtime_entrypoint = build_runtime_entrypoint(entrypoint, resolved_environment)
        spec = JobSpec(
            version=1,
            name=job_id.to_wire(),
            entrypoint=runtime_entrypoint,
            resources=resources,
            environment=resolved_environment,
            bundle_id="",
            scheduling_timeout=scheduling_timeout,
            ports=tuple(ports or ()),
            max_task_failures=max_task_failures,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            constraints=tuple(constraints or ()),
            coscheduling=coscheduling,
            replicas=replicas,
            timeout=timeout,
            fail_if_exists=False,
            preemption_policy=preemption_policy,
            existing_job_policy=existing_job_policy,
            priority_band=priority_band,
            task_image=task_image or "",
            submit_argv=tuple(submit_argv or ()),
            client_revision_date=client_revision_date(),
            container_profile=container_profile,
        )

        try:
            identity = self.submit_job(spec)
        except ConnectError as e:
            if e.code == Code.ALREADY_EXISTS:
                raise JobAlreadyExists(str(e)) from e
            raise

        return Job(self, identity)

    def submit_job(self, spec: JobSpec, *, bundle: bytes | None = None) -> JobIdentity:
        return self._cluster_client.submit_job(spec, bundle=bundle)

    def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]:
        return self._cluster_client.list_jobs(query)

    def describe_job(self, key: ResourceKey) -> JobDetail:
        return self._cluster_client.describe_job(key)

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]:
        return self._cluster_client.list_tasks(query)

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        return self._cluster_client.describe_task(key)

    def describe_tasks(self, keys: Sequence[ResourceKey]) -> tuple[TaskDetail, ...]:
        return self._cluster_client.describe_tasks(keys)

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail:
        return self._cluster_client.describe_attempt(locator)

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        return self._cluster_client.list_nodes(query)

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        return self._cluster_client.describe_node(locator)

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]:
        return self._cluster_client.list_slices(query)

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        return self._cluster_client.describe_slice(locator)

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]:
        return self._cluster_client.list_endpoints(query)

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        return self._cluster_client.describe_endpoint(key)

    def describe_endpoints(self, keys: Sequence[ResourceKey]) -> tuple[EndpointDetail, ...]:
        return self._cluster_client.describe_endpoints(keys)

    def resolve_endpoints(self, name: str) -> tuple[EndpointDetail, ...]:
        return self._cluster_client.resolve_endpoints(name)

    def mint_endpoint_token(self, key: ResourceKey, *, ttl: Duration) -> EndpointToken:
        return self._cluster_client.mint_endpoint_token(key, ttl=ttl)

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        return self._cluster_client.list_activity(query)

    def fetch_job_logs(self, identity: JobIdentity, query: LogQuery = LogQuery()) -> LogPage:
        return self._cluster_client.fetch_job_logs(identity, query)

    def fetch_task_logs(self, identity: TaskIdentity, query: LogQuery = LogQuery()) -> LogPage:
        return self._cluster_client.fetch_task_logs(identity, query)

    def fetch_attempt_logs(self, identity: AttemptIdentity, query: LogQuery = LogQuery()) -> LogPage:
        return self._cluster_client.fetch_attempt_logs(identity, query)

    def stream_job_logs(
        self,
        identity: JobIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[LogEntry]:
        return self._cluster_client.stream_job_logs(identity, query)

    def stream_task_logs(
        self,
        identity: TaskIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[LogEntry]:
        return self._cluster_client.stream_task_logs(identity, query)

    def stream_attempt_logs(
        self,
        identity: AttemptIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[LogEntry]:
        return self._cluster_client.stream_attempt_logs(identity, query)

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt:
        return self._cluster_client.cancel_job(identity, idempotency_key=idempotency_key)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt:
        return self._cluster_client.retry_task(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
        )

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt:
        return self._cluster_client.terminate_attempt(identity, idempotency_key=idempotency_key)

    def get_action_receipt(self, action_id: str) -> ActionReceipt:
        return self._cluster_client.get_action_receipt(action_id)

    def wait_for_action(self, action_id: str, *, timeout: Duration) -> ActionReceipt:
        return self._cluster_client.wait_for_action(action_id, timeout=timeout)

    def exec_attempt(
        self,
        identity: AttemptIdentity,
        *,
        command: Sequence[str],
        timeout: Duration,
    ) -> ExecResult:
        return self._cluster_client.exec_attempt(identity, command=command, timeout=timeout)

    def profile_attempt(
        self,
        identity: AttemptIdentity,
        *,
        profile: ProfileConfiguration,
        duration: Duration,
    ) -> ProfileResult:
        return self._cluster_client.profile_attempt(identity, profile=profile, duration=duration)

    def current_job(self, job_id: JobName) -> Job:
        """Resolve the current Job incarnation for a wire ID."""
        query = JobQuery(job_id_prefix=job_id.to_wire(), page_size=_RESOURCE_PAGE_SIZE)
        while True:
            page = self.list_jobs(query)
            match = next(
                (summary for summary in page.items if summary.identity.key.resource_id == job_id.to_wire()),
                None,
            )
            if match is not None:
                return Job(self, match.identity)
            if page.next_page_token is None:
                raise ConnectError(Code.NOT_FOUND, f"Job {job_id} not found")
            query = JobQuery(
                job_id_prefix=job_id.to_wire(),
                page_size=_RESOURCE_PAGE_SIZE,
                page_token=page.next_page_token,
            )

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
