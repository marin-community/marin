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

"""Iris backend for fray v2.

Wraps iris.client.IrisClient to implement the fray v2 Client protocol.
Handles type conversion between fray v2 types and Iris types, actor hosting
via submitted jobs, and deferred actor handle resolution.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iris.cluster.types import Constraint, Entrypoint as IrisEntrypoint, EnvironmentSpec, ResourceSpec
    from iris.rpc import cluster_pb2

from fray.v2.actor import ActorFuture, ActorGroup, ActorHandle, FutureActorFuture
from fray.v2.types import (
    CpuConfig,
    DeviceConfig,
    Entrypoint as Entrypoint_v2,
    EnvironmentConfig,
    GpuConfig,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
)
from iris.client.client import IrisClient as IrisClientLib, Job as IrisJob

logger = logging.getLogger(__name__)

# Shared thread pool for async actor RPC calls
_shared_executor: ThreadPoolExecutor | None = None


def _get_shared_executor() -> ThreadPoolExecutor:
    global _shared_executor
    if _shared_executor is None:
        _shared_executor = ThreadPoolExecutor(max_workers=32)
    return _shared_executor


# ---------------------------------------------------------------------------
# Type conversion: fray v2 → Iris
# ---------------------------------------------------------------------------


def _convert_device(device: DeviceConfig) -> cluster_pb2.DeviceConfig | None:
    """Convert fray v2 DeviceConfig to Iris protobuf DeviceConfig."""
    from iris.cluster.types import tpu_device
    from iris.rpc import cluster_pb2

    if isinstance(device, CpuConfig):
        return None
    elif isinstance(device, TpuConfig):
        return tpu_device(device.variant)
    elif isinstance(device, GpuConfig):
        gpu = cluster_pb2.GpuDevice(variant=device.variant, count=device.count)
        return cluster_pb2.DeviceConfig(gpu=gpu)
    raise ValueError(f"Unknown device config type: {type(device)}")


def convert_resources(resources: ResourceConfig) -> ResourceSpec:
    """Convert fray v2 ResourceConfig to Iris ResourceSpec.

    This is the primary type bridge between fray v2 and Iris. The mapping is:
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
        regions=resources.regions,
    )


def convert_constraints(resources: ResourceConfig) -> list[Constraint]:
    """Build Iris scheduling constraints from fray v2 ResourceConfig."""
    from iris.cluster.types import preemptible_constraint

    constraints: list[Constraint] = []
    if not resources.preemptible:
        constraints.append(preemptible_constraint(False))
    return constraints


def convert_entrypoint(entrypoint: Entrypoint_v2) -> IrisEntrypoint:
    """Convert fray v2 Entrypoint to Iris Entrypoint."""
    from iris.cluster.types import Entrypoint as IrisEntrypoint

    if entrypoint.callable_entrypoint is not None:
        ce = entrypoint.callable_entrypoint
        return IrisEntrypoint.from_callable(ce.callable, *ce.args, **ce.kwargs)
    elif entrypoint.binary_entrypoint is not None:
        be = entrypoint.binary_entrypoint
        return IrisEntrypoint.from_command(be.command, *be.args)
    raise ValueError("Entrypoint must have either callable_entrypoint or binary_entrypoint")


def convert_environment(env: EnvironmentConfig | None) -> EnvironmentSpec | None:
    """Convert fray v2 EnvironmentConfig to Iris EnvironmentSpec."""
    if env is None:
        return None
    from iris.cluster.types import EnvironmentSpec

    return EnvironmentSpec(
        pip_packages=list(env.pip_packages),
        env_vars=dict(env.env_vars),
        extras=list(env.extras),
    )


# ---------------------------------------------------------------------------
# Job status mapping: Iris → fray v2
# ---------------------------------------------------------------------------


def map_iris_job_state(iris_state: int) -> JobStatus:
    """Map Iris protobuf JobState enum to fray v2 JobStatus."""
    from iris.rpc import cluster_pb2

    _STATE_MAP = {
        cluster_pb2.JOB_STATE_PENDING: JobStatus.PENDING,
        cluster_pb2.JOB_STATE_RUNNING: JobStatus.RUNNING,
        cluster_pb2.JOB_STATE_SUCCEEDED: JobStatus.SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED: JobStatus.FAILED,
        cluster_pb2.JOB_STATE_KILLED: JobStatus.STOPPED,
        cluster_pb2.JOB_STATE_WORKER_FAILED: JobStatus.FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE: JobStatus.FAILED,
    }
    return _STATE_MAP.get(iris_state, JobStatus.PENDING)


# ---------------------------------------------------------------------------
# IrisJobHandle
# ---------------------------------------------------------------------------


class IrisJobHandle:
    """JobHandle wrapping an iris.client.Job."""

    def __init__(self, job: IrisJob):
        self._job = job

    @property
    def job_id(self) -> str:
        return str(self._job.job_id)

    def status(self) -> JobStatus:
        iris_status = self._job.status()
        return map_iris_job_state(iris_status.state)

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        effective_timeout = timeout if timeout is not None else 86400.0
        try:
            self._job.wait(timeout=effective_timeout, raise_on_failure=raise_on_failure)
        except Exception:
            if raise_on_failure:
                raise
        return self.status()

    def terminate(self) -> None:
        self._job.terminate()


# ---------------------------------------------------------------------------
# Actor hosting entrypoint
# ---------------------------------------------------------------------------


def _get_host_ip() -> str:
    """Get the IP address that other hosts can reach this machine on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _host_actor(actor_class: type, args: tuple, kwargs: dict, name: str) -> None:
    """Entrypoint for actor-hosting Iris jobs.

    Instantiates the actor class, creates an ActorServer, registers the
    endpoint for discovery, and blocks until the job is terminated.
    """
    from iris.actor.server import ActorServer
    from iris.client.client import iris_ctx

    ctx = iris_ctx()

    instance = actor_class(*args, **kwargs)

    server = ActorServer(host="0.0.0.0", port=ctx.get_port("actor"))
    server.register(name, instance)
    server.serve_background()

    address = f"{_get_host_ip()}:{server._actual_port}"
    ctx.registry.register(name, address)

    # Block forever — job termination kills the process
    threading.Event().wait()


# ---------------------------------------------------------------------------
# IrisActorHandle
# ---------------------------------------------------------------------------


class IrisActorHandle:
    """Handle to an Iris-hosted actor.

    Deferred: created with just an actor name. On first use, resolves the
    name via iris_ctx().resolver to find the actor's address.

    Picklable: serializes as (actor_name, job_id) and reconstructs from the current
    IrisContext on the receiving end. This works because child jobs inherit
    the parent's namespace.
    """

    def __init__(
        self,
        actor_name: str,
        client: Any = None,
        job_id: str | None = None,
        iris_client: IrisClientLib | None = None,
        controller_address: str | None = None,
    ):
        self._actor_name = actor_name
        self._client = client
        self._job_id = job_id
        self._iris_client = iris_client
        self._controller_address = controller_address

    def _resolve(self) -> Any:
        if self._client is None:
            from iris.actor.client import ActorClient
            from iris.client.client import get_iris_ctx

            # If we have an explicit IrisClient and job_id, use that (for external calls)
            if self._iris_client is not None and self._job_id is not None:
                resolver = self._iris_client.resolver_for_job(self._job_id)
            else:
                # Try the context's resolver first (works when called within a job
                # that has iris context properly set up via ContextVar inheritance).
                ctx = get_iris_ctx()
                if ctx is not None and self._job_id is not None and ctx.client is not None:
                    resolver = ctx.client.resolver_for_job(self._job_id)
                elif ctx is not None and ctx.client is not None:
                    resolver = ctx.resolver
                elif self._controller_address is not None and self._job_id is not None:
                    # Fallback: create a lightweight client from the serialized
                    # controller address. This handles the case where the handle
                    # was deserialized in a thread that doesn't inherit the iris
                    # ContextVar (e.g. ActorServer RPC dispatch threads).
                    fallback_client = IrisClientLib.remote(self._controller_address)
                    resolver = fallback_client.resolver_for_job(self._job_id)
                else:
                    raise RuntimeError(
                        f"Cannot resolve actor '{self._actor_name}': no iris context, "
                        f"no iris_client, and no controller_address available"
                    )
            self._client = ActorClient(resolver, self._actor_name)
        return self._client

    def __getattr__(self, method_name: str) -> _IrisActorMethod:
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        return _IrisActorMethod(self, method_name)

    def __getstate__(self) -> dict:
        controller_address = self._controller_address
        if controller_address is None and self._iris_client is not None:
            # Extract controller address from the IrisClient so the handle
            # can reconstruct a resolver after deserialization.
            controller_address = getattr(self._iris_client._cluster, "_address", None)
        return {
            "actor_name": self._actor_name,
            "job_id": self._job_id,
            "controller_address": controller_address,
        }

    def __setstate__(self, state: dict) -> None:
        self._actor_name = state["actor_name"]
        self._job_id = state.get("job_id")
        self._controller_address = state.get("controller_address")
        self._client = None
        self._iris_client = None


class _IrisActorMethod:
    """Wraps a method on an Iris actor. remote() dispatches via thread pool."""

    def __init__(self, handle: IrisActorHandle, method_name: str):
        self._handle = handle
        self._method = method_name

    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        client = self._handle._resolve()
        executor = _get_shared_executor()
        future = executor.submit(lambda: getattr(client, self._method)(*args, **kwargs))
        return FutureActorFuture(future)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        client = self._handle._resolve()
        return getattr(client, self._method)(*args, **kwargs)


# ---------------------------------------------------------------------------
# IrisActorGroup
# ---------------------------------------------------------------------------


class IrisActorGroup(ActorGroup):
    """ActorGroup that polls the Iris resolver to discover actors as they start.

    Unlike LocalClient's ActorGroup where all actors are ready immediately,
    Iris actors become available asynchronously as their hosting jobs start
    and register endpoints.
    """

    def __init__(
        self,
        name: str,
        count: int,
        jobs: list[IrisJobHandle],
        iris_client: IrisClientLib,
    ):
        # Start with empty handles — they'll be populated via wait_ready()
        super().__init__(handles=[], jobs=jobs)
        self._name = name
        self._count = count
        self._iris_client = iris_client
        self._discovered_names: set[str] = set()

    def discover_new(self) -> list[ActorHandle]:
        """Probe for newly available actors without blocking.

        Returns only the handles discovered during this call (not previously
        known ones). Call repeatedly to pick up workers as they come online.
        """
        newly_discovered: list[ActorHandle] = []
        for i in range(self._count):
            actor_name = f"{self._name}-{i}"
            if actor_name in self._discovered_names:
                continue
            job = self._jobs[i]
            resolver = self._iris_client.resolver_for_job(job.job_id)
            result = resolver.resolve(actor_name)

            if not result.is_empty:
                self._discovered_names.add(actor_name)
                controller_address = getattr(self._iris_client._cluster, "_address", None)
                handle = IrisActorHandle(
                    actor_name,
                    job_id=job.job_id,
                    iris_client=self._iris_client,
                    controller_address=controller_address,
                )
                self._handles.append(handle)
                newly_discovered.append(handle)
                logger.info(
                    "discover_new: found actor=%s job_id=%s (%d/%d ready)",
                    actor_name,
                    job.job_id,
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
        sleep_secs = 0.5

        while True:
            self.discover_new()

            if len(self._discovered_names) >= target:
                return list(self._handles[:target])

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                for i in range(self._count):
                    actor_name = f"{self._name}-{i}"
                    job = self._jobs[i]
                    all_eps = self._iris_client._cluster.list_endpoints(prefix="")
                    logger.error(
                        "wait_ready TIMEOUT: actor=%s job_id=%s all_endpoints=%s",
                        actor_name,
                        job.job_id,
                        [(ep.name, ep.address, ep.job_id) for ep in all_eps],
                    )
                raise TimeoutError(f"Only {len(self._discovered_names)}/{target} actors ready after {timeout}s")

            time.sleep(sleep_secs)


# ---------------------------------------------------------------------------
# FrayIrisClient
# ---------------------------------------------------------------------------


class FrayIrisClient:
    """Iris cluster backend for fray v2.

    Wraps iris.client.IrisClient to implement the fray v2 Client protocol.
    Jobs are submitted via Iris, actors are hosted as Iris jobs with endpoint
    registration for discovery.
    """

    def __init__(
        self,
        controller_address: str,
        workspace: Path | None = None,
        bundle_gcs_path: str | None = None,
    ):
        logger.info(
            "FrayIrisClient connecting to %s (workspace=%s, bundle_gcs_path=%s)",
            controller_address,
            workspace,
            bundle_gcs_path,
        )
        self._iris = IrisClientLib.remote(controller_address, workspace=workspace, bundle_gcs_path=bundle_gcs_path)

    def submit(self, request: JobRequest) -> IrisJobHandle:
        iris_resources = convert_resources(request.resources)
        iris_entrypoint = convert_entrypoint(request.entrypoint)
        iris_environment = convert_environment(request.environment)
        iris_constraints = convert_constraints(request.resources)

        job = self._iris.submit(
            entrypoint=iris_entrypoint,
            name=request.name,
            resources=iris_resources,
            environment=iris_environment,
            constraints=iris_constraints if iris_constraints else None,
            replicas=request.replicas,
        )
        return IrisJobHandle(job)

    def create_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        resources: ResourceConfig = ResourceConfig(),
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
        **kwargs: Any,
    ) -> IrisActorGroup:
        """Submit N Iris jobs, each hosting an instance of actor_class."""
        from iris.cluster.types import Entrypoint as IrisEntrypoint

        iris_resources = convert_resources(resources)
        iris_environment = convert_environment(None)
        iris_constraints = convert_constraints(resources)

        jobs: list[IrisJobHandle] = []
        for i in range(count):
            actor_name = f"{name}-{i}"
            entrypoint = IrisEntrypoint.from_callable(_host_actor, actor_class, args, kwargs, actor_name)
            job = self._iris.submit(
                entrypoint=entrypoint,
                name=actor_name,
                resources=iris_resources,
                environment=iris_environment,
                ports=["actor"],
                constraints=iris_constraints if iris_constraints else None,
            )
            jobs.append(IrisJobHandle(job))

        return IrisActorGroup(
            name=name,
            count=count,
            jobs=jobs,
            iris_client=self._iris,
        )

    def shutdown(self, wait: bool = True) -> None:
        self._iris.shutdown(wait=wait)
