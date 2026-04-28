# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris backend for fray.

Wraps iris.client.IrisClient to implement the fray Client protocol.
Handles type conversion between fray types and Iris types, actor hosting
via submitted jobs, and deferred actor handle resolution.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import cloudpickle

from iris.actor.client import ActorClient
from iris.actor.server import ActorServer
from iris.client.client import IrisClient as IrisClientLib
from iris.client.client import Job as IrisJob
from iris.client.client import JobAlreadyExists as IrisJobAlreadyExists
from iris.client.client import get_iris_ctx, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import (
    Constraint,
    device_variant_constraint,
    preemptible_constraint,
    region_constraint,
    zone_constraint,
)
from iris.cluster.types import CoschedulingConfig, EnvironmentSpec, ResourceSpec, is_job_finished
from iris.cluster.types import Entrypoint as IrisEntrypoint
from iris.rpc import job_pb2
from rigging.timing import ExponentialBackoff

from fray.actor import (
    ActorContext,
    ActorFuture,
    ActorHandle,
    HostedActor,
    _reset_current_actor,
    _set_current_actor,
)
from fray.client import JobAlreadyExists as FrayJobAlreadyExists
from fray.types import (
    ActorConfig,
    CpuConfig,
    DeviceConfig,
    EnvironmentConfig,
    GpuConfig,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
)
from fray.types import (
    Entrypoint as FrayEntrypoint,
)

logger = logging.getLogger(__name__)


def resolve_coscheduling(device: DeviceConfig, replicas: int) -> CoschedulingConfig | None:
    """Determine coscheduling config for multi-host jobs."""
    if replicas <= 1:
        return None
    if isinstance(device, TpuConfig):
        if device.vm_count() <= 1:
            return None
        return CoschedulingConfig(group_by="tpu-name")
    if isinstance(device, GpuConfig):
        return CoschedulingConfig(group_by="pool")
    return None


def _convert_device(device: DeviceConfig) -> job_pb2.DeviceConfig | None:
    """Convert fray DeviceConfig to Iris protobuf DeviceConfig."""
    from iris.cluster.types import tpu_device

    if isinstance(device, CpuConfig):
        return None
    elif isinstance(device, TpuConfig):
        return tpu_device(device.variant)
    elif isinstance(device, GpuConfig):
        gpu = job_pb2.GpuDevice(variant=device.variant, count=device.count)
        return job_pb2.DeviceConfig(gpu=gpu)
    raise ValueError(f"Unknown device config type: {type(device)}")


def convert_resources(resources: ResourceConfig) -> ResourceSpec:
    """Convert fray ResourceConfig to Iris ResourceSpec.

    This is the primary type bridge between fray and Iris. The mapping is:
      fray cpu       → Iris cpu
      fray ram       → Iris memory
      fray disk      → Iris disk
      fray device    → Iris device (TPU via tpu_device(), GPU via GpuDevice)
    Replicas are passed separately to iris client.submit().
    """
    from iris.cluster.types import ResourceSpec

    return ResourceSpec(
        cpu=resources.cpu,
        memory=resources.ram,
        disk=resources.disk,
        device=_convert_device(resources.device),
    )


def convert_constraints(resources: ResourceConfig) -> list[Constraint]:
    """Build Iris scheduling constraints from fray ResourceConfig."""
    constraints: list[Constraint] = []
    if not resources.preemptible:
        constraints.append(preemptible_constraint(False))
    if resources.regions:
        constraints.append(region_constraint(resources.regions))
    if resources.zone:
        constraints.append(zone_constraint(resources.zone))
    if resources.device_alternatives:
        if isinstance(resources.device, (TpuConfig, GpuConfig)):
            all_variants = [resources.device.variant, *resources.device_alternatives]
            constraints.append(device_variant_constraint(all_variants))
    return constraints


def convert_entrypoint(entrypoint: FrayEntrypoint) -> IrisEntrypoint:
    """Convert fray Entrypoint to Iris Entrypoint."""
    from iris.cluster.types import Entrypoint as IrisEntrypoint

    if entrypoint.callable_entrypoint is not None:
        ce = entrypoint.callable_entrypoint
        return IrisEntrypoint.from_callable(ce.callable, *ce.args, **ce.kwargs)
    elif entrypoint.binary_entrypoint is not None:
        be = entrypoint.binary_entrypoint
        return IrisEntrypoint.from_command(be.command, *be.args)
    raise ValueError("Entrypoint must have either callable_entrypoint or binary_entrypoint")


def convert_environment(env: EnvironmentConfig | None, device: DeviceConfig | None = None) -> EnvironmentSpec | None:
    """Convert fray EnvironmentConfig to Iris EnvironmentSpec."""
    env_vars = dict(env.env_vars) if env is not None else {}
    if device is not None:
        for key, value in device.default_env_vars().items():
            env_vars.setdefault(key, value)
    if env is None and not env_vars:
        return None
    from iris.cluster.types import EnvironmentSpec

    return EnvironmentSpec(
        pip_packages=list(env.pip_packages) if env is not None else [],
        env_vars=env_vars,
        extras=list(env.extras) if env is not None else [],
    )


def map_iris_job_state(iris_state: int) -> JobStatus:
    """Map Iris protobuf JobState enum to fray JobStatus."""

    _STATE_MAP = {
        job_pb2.JOB_STATE_PENDING: JobStatus.PENDING,
        job_pb2.JOB_STATE_RUNNING: JobStatus.RUNNING,
        job_pb2.JOB_STATE_SUCCEEDED: JobStatus.SUCCEEDED,
        job_pb2.JOB_STATE_FAILED: JobStatus.FAILED,
        job_pb2.JOB_STATE_KILLED: JobStatus.STOPPED,
        job_pb2.JOB_STATE_WORKER_FAILED: JobStatus.FAILED,
        job_pb2.JOB_STATE_UNSCHEDULABLE: JobStatus.FAILED,
    }
    return _STATE_MAP.get(iris_state, JobStatus.PENDING)


class IrisJobHandle:
    """JobHandle wrapping an iris.client.Job."""

    def __init__(self, job: IrisJob):
        self._job = job

    @property
    def job_id(self) -> str:
        return str(self._job.job_id)

    def status(self) -> JobStatus:
        iris_state = self._job.state_only()
        return map_iris_job_state(iris_state)

    def wait(
        self, timeout: float | None = None, *, raise_on_failure: bool = True, stream_logs: bool = False
    ) -> JobStatus:
        # Iris client requires a numeric timeout. When None (wait indefinitely),
        # use ~5 years so the caller is never surprised by a silent timeout.
        effective_timeout = timeout if timeout is not None else 86400.0 * 365 * 5
        try:
            self._job.wait(timeout=effective_timeout, raise_on_failure=raise_on_failure, stream_logs=stream_logs)
        except Exception:
            if raise_on_failure:
                raise
            logger.warning("Job %s failed with exception (raise_on_failure=False)", self.job_id, exc_info=True)
        return self.status()

    def terminate(self) -> None:
        self._job.terminate()


def _host_actor(actor_class: type, args: tuple, kwargs: dict, name_prefix: str) -> None:
    """Entrypoint for actor-hosting Iris jobs.

    Instantiates the actor class, creates an ActorServer, registers the
    endpoint for discovery, and blocks until the job is terminated.

    For multi-replica jobs, each replica gets a unique actor name based on
    its task index. Uses absolute endpoint names: "/{job_id}/{name_prefix}-{task_index}".
    """

    ctx = iris_ctx()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("_host_actor must run inside an Iris job but get_job_info() returned None")

    # Absolute endpoint name - bypasses namespace prefix in resolver
    # JobName.__str__ already includes leading slash
    actor_name = f"{ctx.job_id}/{name_prefix}-{job_info.task_index}"
    logger.info(f"Starting actor: {actor_name} (job_id={ctx.job_id})")

    # Shutdown event: the actor sets it when ready to exit,
    # unblocking the wait below.
    shutdown_event = threading.Event()

    # Create handle BEFORE instance so actor can access it during __init__
    handle = IrisActorHandle(actor_name)
    actor_ctx = ActorContext(
        handle=handle, index=job_info.task_index, group_name=name_prefix, shutdown_event=shutdown_event
    )
    token = _set_current_actor(actor_ctx)
    try:
        instance = actor_class(*args, **kwargs)
    finally:
        _reset_current_actor(token)

    server = ActorServer(host="0.0.0.0", port=ctx.get_port("actor"))
    server.register(actor_name, instance)
    actual_port = server.serve_background()

    advertise_host = job_info.advertise_host
    # XXX: this should be handled by the actor server?
    address = f"http://{advertise_host}:{actual_port}"
    logger.info(f"Registering endpoint: {actor_name} -> {address}")
    ctx.registry.register(actor_name, address)
    logger.info(f"Actor {actor_name} ready and listening")

    # Block until the actor signals shutdown via shutdown_event
    shutdown_event.wait()
    logger.info(f"Actor {actor_name} shutting down")
    server.stop()


class IrisActorHandle:
    """Handle to an Iris-hosted actor. Resolves via iris_ctx()."""

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name
        self._client: Any = None  # Lazily resolved ActorClient

    def __getstate__(self) -> dict:
        # Only serialize the endpoint name - client is lazily resolved
        return {"endpoint_name": self._endpoint_name}

    def __setstate__(self, state: dict) -> None:
        self._endpoint_name = state["endpoint_name"]
        self._client = None

    def _resolve(self) -> Any:
        """Resolve endpoint to ActorClient via IrisContext."""
        if self._client is None:
            ctx = get_iris_ctx()
            if ctx is None:
                raise RuntimeError(
                    "IrisActorHandle._resolve() requires IrisContext. "
                    "Call from within an Iris job or set context via iris_ctx_scope()."
                )
            self._client = ActorClient(ctx.resolver, self._endpoint_name)
        return self._client

    def __getattr__(self, method_name: str) -> _IrisActorMethod:
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        return _IrisActorMethod(self, method_name)


class OperationFuture:
    """Polling-based future backed by an Iris long-running operation.

    Satisfies the ``ActorFuture`` protocol. Each call to ``result()`` polls
    the server via short ``GetOperation`` RPCs until the operation completes,
    fails, or the caller's timeout expires.
    """

    def __init__(self, client: ActorClient, operation_id: str, poll_interval: float = 1.0):
        self._client = client
        self._op_id = operation_id
        self._poll_interval = poll_interval

    def result(self, timeout: float | None = None) -> Any:
        from iris.rpc import actor_pb2

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            op = self._client.poll_operation_status(self._op_id)

            if op.state == actor_pb2.Operation.SUCCEEDED:
                return cloudpickle.loads(op.serialized_result)

            if op.state == actor_pb2.Operation.FAILED:
                if op.error.serialized_exception:
                    raise cloudpickle.loads(op.error.serialized_exception)
                raise RuntimeError(f"{op.error.error_type}: {op.error.message}")

            if op.state == actor_pb2.Operation.CANCELLED:
                raise RuntimeError(f"Operation {self._op_id} was cancelled")

            if deadline is not None and time.monotonic() >= deadline:
                self._client.cancel_operation(self._op_id)
                raise TimeoutError(f"Operation {self._op_id} timed out after {timeout}s")

            time.sleep(self._poll_interval)


class _ThreadFuture:
    """Future backed by a daemon thread running a direct Call RPC.

    Unlike OperationFuture, this holds a single HTTP connection for the
    call duration and does not poll. Suitable for short RPCs where the
    overhead of StartOperation + GetOperation polling is unnecessary.
    """

    def __init__(self, fn: Any, args: tuple, kwargs: dict):
        self._future: Future[Any] = Future()

        def run() -> None:
            try:
                self._future.set_result(fn(*args, **kwargs))
            except Exception as e:
                self._future.set_exception(e)

        threading.Thread(target=run, daemon=True).start()

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)


class _IrisActorMethod:
    """Wraps a method on an Iris actor.

    ``remote()`` spawns a thread running a direct ``Call`` RPC — fast
    (single RPC) and suitable for short-lived methods.

    ``submit()`` uses long-running operations: a fast ``StartOperation``
    RPC returns an operation ID, and ``OperationFuture.result()`` polls
    via ``GetOperation``. Use for methods that run for minutes/hours.

    ``__call__()`` uses the blocking ``Call`` RPC synchronously.
    """

    def __init__(self, handle: IrisActorHandle, method_name: str):
        self._handle = handle
        self._method = method_name

    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        client = self._handle._resolve()
        method = getattr(client, self._method)
        return _ThreadFuture(method, args, kwargs)

    def submit(self, *args: Any, **kwargs: Any) -> ActorFuture:
        client = self._handle._resolve()
        op_id = client.start_operation(self._method, *args, **kwargs)
        return OperationFuture(client, op_id)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        client = self._handle._resolve()
        return getattr(client, self._method)(*args, **kwargs)


class IrisActorGroup:
    """ActorGroup that polls the Iris resolver to discover actors as they start."""

    def __init__(self, name: str, count: int, job_id: Any):
        """Args:
        name: Actor name prefix
        count: Number of actors to discover
        job_id: JobId/JobName for the actor job
        """
        self._name = name
        self._count = count
        self._job_id = job_id
        self._handles: list[ActorHandle] = []
        self._discovered_names: set[str] = set()

    def __getstate__(self) -> dict:
        # Serialize only the discovery parameters - discovery state resets on deserialize
        return {
            "name": self._name,
            "count": self._count,
            "job_id": self._job_id,
        }

    def __setstate__(self, state: dict) -> None:
        self._name = state["name"]
        self._count = state["count"]
        self._job_id = state["job_id"]
        self._handles = []
        self._discovered_names = set()

    def _get_client(self) -> IrisClientLib:
        """Get IrisClient from context."""
        from iris.client.client import get_iris_ctx

        ctx = get_iris_ctx()
        if ctx is None or ctx.client is None:
            raise RuntimeError("IrisActorGroup requires IrisContext with client. Set context via iris_ctx_scope().")
        return ctx.client

    @property
    def ready_count(self) -> int:
        """Number of actors that are available for RPC."""
        return len(self._handles)

    def discover_new(self, target: int | None = None) -> list[ActorHandle]:
        """Probe for newly available actors without blocking.

        Returns only the handles discovered during this call (not previously
        known ones). Call repeatedly to pick up workers as they come online.

        Uses a single prefix-match RPC to discover all actors whose endpoint
        names start with ``{job_id}/{name}-``.

        Args:
            target: Stop probing once this many total actors are discovered.
                If None, probes all indices.
        """
        client = self._get_client()
        # Single RPC: prefix match all actors for this group
        # _host_actor registers endpoints as "{job_id}/{name}-{task_index}"
        prefix = f"{self._job_id}/{self._name}-"
        endpoints = client._cluster_client.list_endpoints(prefix=prefix, exact=False)

        newly_discovered: list[ActorHandle] = []
        for ep in endpoints:
            if target is not None and len(self._discovered_names) >= target:
                break
            if ep.name in self._discovered_names:
                continue
            self._discovered_names.add(ep.name)
            handle = IrisActorHandle(ep.name)
            self._handles.append(handle)
            newly_discovered.append(handle)
            logger.info(
                "discover_new: found actor=%s job_id=%s (%d/%d ready)",
                ep.name,
                self._job_id,
                len(self._discovered_names),
                self._count,
            )

        return newly_discovered

    def wait_ready(self, count: int | None = None, timeout: float = 900.0) -> list[ActorHandle]:
        """Block until `count` actors are discoverable via the resolver.

        With count=1 this returns as soon as the first worker is available,
        allowing the caller to start work immediately and discover more
        workers later via discover_new().
        """
        target = count if count is not None else self._count
        start = time.monotonic()
        backoff = ExponentialBackoff(initial=0.1, maximum=5.0)

        while True:
            self.discover_new(target=target)

            if len(self._discovered_names) >= target:
                return list(self._handles[:target])

            # Fail fast if the underlying job has terminated (e.g. crash, OOM,
            # missing interpreter). Without this check we'd spin for the full
            # timeout waiting for endpoints that will never appear. Use the
            # lightweight state-only RPC and fetch the full status only when we
            # actually need the error message.
            client = self._get_client()
            state = client.job_state(self._job_id)
            if is_job_finished(state):
                error = client.status(self._job_id).error or "unknown error"
                raise RuntimeError(
                    f"Actor job {self._job_id} finished before all actors registered "
                    f"({len(self._discovered_names)}/{target} ready). "
                    f"Job state={state}, error={error}"
                )

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Only {len(self._discovered_names)}/{target} actors ready after {timeout}s")

            time.sleep(backoff.next_interval())

    def is_done(self) -> bool:
        """Return True if the Iris worker job has permanently terminated."""
        client = self._get_client()
        return is_job_finished(client.job_state(self._job_id))

    def shutdown(self) -> None:
        """Terminate the actor job."""
        client = self._get_client()
        client.terminate(self._job_id)


class FrayIrisClient:
    """Iris cluster backend for fray.

    Wraps iris.client.IrisClient to implement the fray Client protocol.
    Jobs are submitted via Iris, actors are hosted as Iris jobs with endpoint
    registration for discovery.
    """

    def __init__(
        self,
        controller_address: str,
        workspace: Path | None = None,
        bundle_id: str | None = None,
    ):
        logger.info(
            "FrayIrisClient connecting to %s (workspace=%s, bundle_id=%s)",
            controller_address,
            workspace,
            bundle_id,
        )
        self._iris = IrisClientLib.remote(controller_address, workspace=workspace, bundle_id=bundle_id)

    @staticmethod
    def from_iris_client(iris_client: IrisClientLib) -> FrayIrisClient:
        """Create a FrayIrisClient by wrapping an existing IrisClient.

        This avoids creating a new connection when we already have an IrisClient
        from the context (e.g., when running inside an Iris task).
        """
        instance = cast(FrayIrisClient, object.__new__(FrayIrisClient))
        instance._iris = iris_client
        return instance

    def submit(self, request: JobRequest, adopt_existing: bool = True) -> IrisJobHandle:
        iris_resources = convert_resources(request.resources)
        iris_entrypoint = convert_entrypoint(request.entrypoint)
        iris_environment = convert_environment(request.environment, request.resources.device)
        iris_constraints = convert_constraints(request.resources)

        replicas = request.replicas or 1
        coscheduling = resolve_coscheduling(request.resources.device, replicas)

        policy = job_pb2.EXISTING_JOB_POLICY_KEEP if adopt_existing else job_pb2.EXISTING_JOB_POLICY_UNSPECIFIED
        try:
            job = self._iris.submit(
                entrypoint=iris_entrypoint,
                name=request.name,
                resources=iris_resources,
                environment=iris_environment,
                constraints=iris_constraints if iris_constraints else None,
                coscheduling=coscheduling,
                replicas=replicas,
                max_retries_failure=request.max_retries_failure,
                max_retries_preemption=request.max_retries_preemption,
                existing_job_policy=policy,
                task_image=request.resources.image,
            )
        except IrisJobAlreadyExists as e:
            raise FrayJobAlreadyExists(request.name) from e
        return IrisJobHandle(job)

    def host_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        actor_config: ActorConfig = ActorConfig(),
        **kwargs: Any,
    ) -> HostedActor:
        """Host an actor in the current process with Iris RPC serving."""
        ctx = iris_ctx()
        job_info = get_job_info()
        if job_info is None:
            raise RuntimeError("host_actor requires an Iris job context")

        actor_name = f"{ctx.job_id}/{name}-0"
        handle = IrisActorHandle(actor_name)
        actor_ctx = ActorContext(handle=handle, index=0, group_name=name)
        token = _set_current_actor(actor_ctx)
        try:
            instance = actor_class(*args, **kwargs)
        finally:
            _reset_current_actor(token)

        server = ActorServer(host="0.0.0.0", port=0)
        server.register(actor_name, instance)
        actual_port = server.serve_background()

        address = f"http://{job_info.advertise_host}:{actual_port}"
        logger.info("host_actor: registered %s -> %s", actor_name, address)
        ctx.registry.register(actor_name, address)

        return HostedActor(handle, stop=server.stop)

    def create_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        resources: ResourceConfig = ResourceConfig(),
        actor_config: ActorConfig = ActorConfig(),
        **kwargs: Any,
    ) -> ActorHandle:
        group = self.create_actor_group(actor_class, *args, name=name, count=1, resources=resources, **kwargs)
        return group.wait_ready()[0]

    def create_actor_group(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        count: int,
        resources: ResourceConfig = ResourceConfig(),
        actor_config: ActorConfig = ActorConfig(),
        **kwargs: Any,
    ) -> IrisActorGroup:
        """Submit a single Iris job with N replicas, each hosting an instance of actor_class.

        Uses Iris's multi-replica job feature instead of creating N separate jobs,
        which improves networking and reduces job overhead.
        """
        from iris.cluster.types import Entrypoint as IrisEntrypoint

        iris_resources = convert_resources(resources)
        iris_constraints = convert_constraints(resources)
        iris_environment = convert_environment(None, device=resources.device)

        coscheduling = resolve_coscheduling(resources.device, count)

        # Create a single job with N replicas
        # Each replica will run _host_actor with a unique task-based actor name
        entrypoint = IrisEntrypoint.from_callable(_host_actor, actor_class, args, kwargs, name)

        retry_kwargs: dict[str, Any] = {}
        if actor_config.max_task_retries is not None:
            retry_kwargs["max_retries_failure"] = actor_config.max_task_retries

        job = self._iris.submit(
            entrypoint=entrypoint,
            name=name,
            resources=iris_resources,
            environment=iris_environment,
            ports=["actor"],
            constraints=iris_constraints if iris_constraints else None,
            coscheduling=coscheduling,
            replicas=count,  # Create N replicas in a single job
            task_image=resources.image,
            **retry_kwargs,
        )

        return IrisActorGroup(
            name=name,
            count=count,
            job_id=job.job_id,
        )

    def shutdown(self, wait: bool = True) -> None:
        self._iris.shutdown(wait=wait)
