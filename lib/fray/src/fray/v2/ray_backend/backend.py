# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray backend for fray v2 Client protocol."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, cast

import humanfriendly
import ray
from ray.job_submission import JobStatus as RayJobStatus
from ray.job_submission import JobSubmissionClient

from fray.v2.actor import ActorContext, ActorFuture, ActorGroup, ActorHandle, _reset_current_actor, _set_current_actor
from fray.v2.ray_backend.deps import build_python_path, build_runtime_env_for_packages
from fray.v2.ray_backend.tpu import run_on_pod_ray
from fray.v2.types import (
    CpuConfig,
    GpuConfig,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
    create_environment,
    get_tpu_topology,
)

logger = logging.getLogger(__name__)


def _convert_ray_status(ray_status: RayJobStatus) -> JobStatus:
    mapping = {
        RayJobStatus.PENDING: JobStatus.PENDING,
        RayJobStatus.RUNNING: JobStatus.RUNNING,
        RayJobStatus.SUCCEEDED: JobStatus.SUCCEEDED,
        RayJobStatus.FAILED: JobStatus.FAILED,
        RayJobStatus.STOPPED: JobStatus.STOPPED,
    }
    return mapping.get(ray_status, JobStatus.FAILED)


class RayJobHandle:
    """Job handle wrapping either a Ray ObjectRef or a Ray job submission ID.

    Ray has two job models: remote tasks (ObjectRef) and submitted jobs
    (submission ID via JobSubmissionClient). TPU and callable jobs use
    ObjectRef; binary jobs use submission ID.
    """

    def __init__(
        self,
        job_id: str,
        *,
        ref: ray.ObjectRef | None = None,
        submission_id: str | None = None,
        dashboard_address: str | None = None,
    ):
        self._job_id = job_id
        self._ref = ref
        self._submission_id = submission_id
        self._dashboard_address = dashboard_address

    @property
    def job_id(self) -> str:
        return self._job_id

    def status(self) -> JobStatus:
        if self._ref is not None:
            return self._poll_ref()
        return self._poll_submission()

    def _poll_ref(self) -> JobStatus:
        ready, _ = ray.wait([self._ref], timeout=0)
        if not ready:
            return JobStatus.RUNNING
        try:
            ray.get(self._ref)
            return JobStatus.SUCCEEDED
        except Exception:
            logger.error("Job %s failed:", self._job_id, exc_info=True)
            return JobStatus.FAILED

    def _poll_submission(self) -> JobStatus:
        client = JobSubmissionClient(self._dashboard_address)
        info = client.get_job_info(self._submission_id)
        return _convert_ray_status(info.status)

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        """Block until the job completes."""
        if self._ref is not None:
            return self._wait_ref(timeout, raise_on_failure)
        return self._wait_submission(timeout, raise_on_failure)

    def _wait_ref(self, timeout: float | None, raise_on_failure: bool) -> JobStatus:
        try:
            ray.get(self._ref, timeout=timeout)
        except Exception:
            if raise_on_failure:
                raise
            logger.error("Job %s failed:", self._job_id, exc_info=True)
        return self.status()

    def _wait_submission(self, timeout: float | None, raise_on_failure: bool) -> JobStatus:
        start = time.monotonic()
        sleep_secs = 0.5
        while True:
            s = self.status()
            if JobStatus.finished(s):
                if raise_on_failure and s in (JobStatus.FAILED, JobStatus.STOPPED):
                    raise RuntimeError(f"Job {self._job_id} finished with status {s}")
                return s
            if timeout is not None and (time.monotonic() - start) > timeout:
                raise TimeoutError(f"Job {self._job_id} timed out after {timeout}s")
            time.sleep(sleep_secs)
            sleep_secs = min(sleep_secs * 1.5, 5.0)

    def terminate(self) -> None:
        if self._ref is not None:
            ray.cancel(self._ref)
            return
        try:
            client = JobSubmissionClient(self._dashboard_address)
            client.stop_job(self._submission_id)
        except Exception as e:
            logger.warning("Failed to stop job %s: %s", self._job_id, e)


def compute_ray_retry_count(request: JobRequest) -> int:
    """Map separate failure/preemption retry counts to a single Ray retry count."""
    return request.max_retries_failure + request.max_retries_preemption


def get_entrypoint_params(request: JobRequest) -> dict[str, Any]:
    """Build entrypoint resource params for Ray job submission (binary jobs)."""
    params: dict[str, Any] = {}

    if request.resources.cpu > 0:
        params["entrypoint_num_cpus"] = float(request.resources.cpu)

    if request.resources.ram:
        params["entrypoint_memory"] = humanfriendly.parse_size(request.resources.ram, binary=True)

    device = request.resources.device
    if isinstance(device, GpuConfig):
        params["entrypoint_num_gpus"] = float(device.count)
    elif isinstance(device, TpuConfig):
        params["entrypoint_resources"] = {
            f"TPU-{device.variant}-head": 1.0,
            "TPU": float(device.chip_count()),
        }

    return params


def build_runtime_env(request: JobRequest) -> dict:
    """Build Ray runtime environment for a job request."""
    environment = request.environment if request.environment else create_environment()

    env_vars = dict(environment.env_vars)
    extras = list(environment.extras)

    if isinstance(request.resources.device, CpuConfig):
        if "JAX_PLATFORMS" in env_vars and env_vars["JAX_PLATFORMS"] != "cpu":
            logger.warning(
                "Found existing JAX_PLATFORMS=%s, overriding for CPU only job.",
                env_vars["JAX_PLATFORMS"],
            )
        env_vars["JAX_PLATFORMS"] = "cpu"
    elif isinstance(request.resources.device, TpuConfig):
        if "tpu" not in extras:
            extras.append("tpu")
        env_vars["JAX_PLATFORMS"] = ""
    elif isinstance(request.resources.device, GpuConfig):
        if "gpu" not in extras:
            extras.append("gpu")
        env_vars["JAX_PLATFORMS"] = ""

    if os.environ.get("MARIN_CI_DISABLE_RUNTIME_ENVS", "").lower() in ("1", "true") or os.environ.get(
        "PYTEST_CURRENT_TEST"
    ):
        logger.info("Skipping runtime_env construction for job: %s", request.name)
        return {"env_vars": env_vars}

    logger.info(
        "Building environment with %s, extras %s for job: %s",
        environment.pip_packages,
        extras,
        request.name,
    )

    if environment.pip_packages or extras:
        runtime_env = build_runtime_env_for_packages(
            extra=extras,
            pip_packages=list(environment.pip_packages),
            env_vars=env_vars,
        )
        runtime_env["working_dir"] = environment.workspace
        runtime_env["excludes"] = [".git", "tests/", "docs/", "**/*.pack"]
        runtime_env["config"] = {"setup_timeout_seconds": 1800}
    else:
        python_path = build_python_path(submodules_dir=os.path.join(environment.workspace, "submodules"))
        python_path = [
            p if os.path.isabs(p) else os.path.join(environment.workspace, p) for p in python_path if p.strip()
        ]
        if "PYTHONPATH" in env_vars:
            python_path.extend([p for p in env_vars["PYTHONPATH"].split(":") if p.strip()])
        env_vars["PYTHONPATH"] = ":".join(python_path)
        runtime_env = {"env_vars": env_vars}

    return runtime_env


class RayClient:
    """Ray backend for fray v2 Client protocol.

    Connects to a Ray cluster and submits jobs as Ray tasks or Ray job
    submissions depending on the entrypoint type:
    - TPU jobs: routed through run_on_pod_ray (gang-scheduled TPU execution)
    - Binary jobs: submitted via JobSubmissionClient
    - Callable jobs: executed as ray.remote tasks
    """

    def __init__(self, address: str = "auto", namespace: str | None = None):
        self._address = os.environ.get("RAY_ADDRESS", "auto") if address == "auto" else address
        if namespace is None:
            self._namespace = f"fray_{uuid.uuid4().hex[:8]}"
        else:
            self._namespace = namespace
        self._dashboard_address = self._get_dashboard_address()
        logger.info("RayClient connected to %s (namespace=%s)", self._address, self._namespace)

    def _get_dashboard_address(self) -> str:
        if ray.is_initialized():
            try:
                ctx = ray.get_runtime_context()
                gcs_address = getattr(ctx, "gcs_address", None)
                if gcs_address is not None:
                    return gcs_address
            except Exception:
                pass
        return self._address

    def submit(self, request: JobRequest) -> RayJobHandle:
        """Submit a job, routing to TPU/binary/callable based on device type."""
        logger.info("Submitting job: %s", request.name)

        if isinstance(request.resources.device, TpuConfig):
            return self._launch_tpu_job(request)

        if request.entrypoint.binary_entrypoint is not None:
            return self._launch_binary_job(request)

        return self._launch_callable_job(request)

    def _launch_binary_job(self, request: JobRequest) -> RayJobHandle:
        entrypoint = request.entrypoint.binary_entrypoint
        runtime_env = build_runtime_env(request)
        entrypoint_cmd = f"{entrypoint.command} {' '.join(entrypoint.args)}"
        entrypoint_params = get_entrypoint_params(request)

        client = JobSubmissionClient(self._dashboard_address)
        submission_timeout_s = float(os.environ.get("FRAY_RAY_JOB_SUBMIT_TIMEOUT_S", "30"))
        deadline = time.time() + submission_timeout_s
        sleep_s = 0.5
        while True:
            try:
                submission_id = client.submit_job(
                    entrypoint=entrypoint_cmd,
                    runtime_env=runtime_env,
                    submission_id=f"{request.name}-{uuid.uuid4()}",
                    metadata={"name": request.name},
                    **entrypoint_params,
                )
                break
            except RuntimeError as e:
                if "No available agent to submit job" not in str(e) or time.time() >= deadline:
                    raise
                logger.info("Ray job agent not ready yet, retrying submit in %.1fs...", sleep_s)
                time.sleep(sleep_s)
                sleep_s = min(5.0, sleep_s * 1.5)

        logger.info("Job submitted with ID: %s", submission_id)
        return RayJobHandle(
            submission_id,
            submission_id=submission_id,
            dashboard_address=self._dashboard_address,
        )

    def _launch_callable_job(self, request: JobRequest) -> RayJobHandle:
        entrypoint = request.entrypoint.callable_entrypoint
        runtime_env = build_runtime_env(request)
        # Strip keys that can only be set at the Job level
        runtime_env = {k: v for k, v in runtime_env.items() if k not in ["working_dir", "excludes", "config"]}

        if isinstance(request.resources.device, GpuConfig):
            num_gpus = request.resources.device.count
        else:
            num_gpus = 0

        remote_fn = ray.remote(num_gpus=num_gpus)(entrypoint.callable)
        ref = remote_fn.options(runtime_env=runtime_env).remote(*entrypoint.args, **entrypoint.kwargs)
        job_id = f"ray-callable-{request.name}-{uuid.uuid4().hex[:8]}"
        return RayJobHandle(job_id, ref=ref)

    def _launch_tpu_job(self, request: JobRequest) -> RayJobHandle:
        callable_ep = request.entrypoint.callable_entrypoint
        assert callable_ep is not None, "TPU jobs require callable entrypoint"

        device = request.resources.device
        runtime_env = build_runtime_env(request)
        runtime_env = {k: v for k, v in runtime_env.items() if k not in ["excludes", "config", "working_dir"]}

        if callable_ep.args or callable_ep.kwargs:
            remote_fn = ray.remote(max_calls=1, runtime_env=runtime_env)(
                lambda: callable_ep.callable(*callable_ep.args, **callable_ep.kwargs)
            )
        else:
            remote_fn = ray.remote(max_calls=1, runtime_env=runtime_env)(callable_ep.callable)

        # Convert replicas (total VMs) back to num_slices for Ray's gang scheduler
        topo = get_tpu_topology(device.variant)
        replicas = request.replicas or 1
        num_slices = max(1, replicas // topo.vm_count)
        object_ref = run_on_pod_ray.remote(
            remote_fn,
            tpu_type=device.variant,
            num_slices=num_slices,
            max_retries_preemption=request.max_retries_preemption,
            max_retries_failure=request.max_retries_failure,
        )

        job_id = f"ray-tpu-{request.name}-{uuid.uuid4().hex[:8]}"
        return RayJobHandle(job_id, ref=object_ref)

    def create_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs: Any,
    ) -> RayActorHandle:
        """Create a single Ray actor and return a handle immediately."""
        group = self.create_actor_group(actor_class, *args, name=name, count=1, resources=resources, **kwargs)
        return group.wait_ready()[0]  # type: ignore[return-value]

    def create_actor_group(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        count: int,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs: Any,
    ) -> ActorGroup:
        """Create N Ray actors named "{name}-0", "{name}-1", ...

        Uses _RayActorHost to wrap actors, enabling them to access their
        own handle via current_actor().handle during __init__.
        """
        ray_options = _actor_ray_options(resources)
        # Don't specify runtime_env - let actors inherit from parent job
        # This prevents rebuilding packages that are already available
        handles: list[RayActorHandle] = []
        for i in range(count):
            actor_name = f"{name}-{i}"
            actor_ref = _RayActorHost.options(name=actor_name, **ray_options).remote(
                actor_class, actor_name, i, name, args, kwargs
            )
            handles.append(RayActorHandle(actor_ref))
        return RayActorGroup(cast(list[ActorHandle], handles))

    def shutdown(self, wait: bool = True) -> None:
        logger.info("RayClient shutdown (namespace=%s)", self._namespace)


def _actor_ray_options(resources: ResourceConfig) -> dict[str, Any]:
    """Build ray.remote().options() kwargs for an actor.

    Maps ResourceConfig fields to Ray scheduling parameters so that
    actors reserve the right amount of CPU/memory and get spread
    across nodes instead of piling onto one.

    preemptible=False pins the actor to the head node via a custom resource.
    max_concurrency>1 enables concurrent method calls (threaded actor).
    """
    options: dict[str, Any] = {
        "num_cpus": resources.cpu,
        "scheduling_strategy": "SPREAD",
    }
    if resources.ram:
        options["memory"] = humanfriendly.parse_size(resources.ram, binary=True)
    if not resources.preemptible:
        options["resources"] = {"head_node": 0.0001}
    if resources.max_concurrency > 1:
        options["max_concurrency"] = resources.max_concurrency
    return options


@ray.remote(enable_task_events=False)
class _RayActorHost:
    """Wrapper that sets up ActorContext before creating the real actor.

    This enables actors to access their own handle via current_actor().handle
    during __init__, even though __init__ runs in a separate process.

    Uses _proxy_call for method dispatch since Ray doesn't support __getattr__
    for dynamic method lookup on actor handles.
    """

    def __init__(
        self,
        actor_class: type,
        actor_name: str,
        actor_index: int,
        group_name: str,
        args: tuple,
        kwargs: dict,
    ):
        # Ensure the root logger is configured in this worker process so that
        # library code using logging.getLogger(__name__).info() is visible.
        # Ray forwards stdout/stderr to the driver, but Python's root logger
        # defaults to WARNING in fresh processes.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
            force=True,
        )

        # Create handle by name - will resolve via ray.get_actor() when used
        handle = RayActorHandle(actor_name)
        ctx = ActorContext(handle=handle, index=actor_index, group_name=group_name)
        token = _set_current_actor(ctx)
        try:
            self._instance = actor_class(*args, **kwargs)
        finally:
            _reset_current_actor(token)

    def _proxy_call(self, method_name: str, args: tuple, kwargs: dict) -> Any:
        """Proxy method calls to the wrapped actor instance."""
        return getattr(self._instance, method_name)(*args, **kwargs)


class RayActorHandle:
    """Handle to a Ray actor. Supports both direct ref and name-based lazy resolution.

    All fray v2 Ray actors are wrapped by _RayActorHost, so method calls go through
    _proxy_call to reach the wrapped instance.
    """

    def __init__(self, actor_ref_or_name: ray.actor.ActorHandle | str):
        # Store under mangled names so __getattr__ doesn't recurse
        if isinstance(actor_ref_or_name, str):
            object.__setattr__(self, "_actor_name", actor_ref_or_name)
            object.__setattr__(self, "_actor_ref", None)
        else:
            object.__setattr__(self, "_actor_name", None)
            object.__setattr__(self, "_actor_ref", actor_ref_or_name)

    def _resolve(self) -> ray.actor.ActorHandle:
        """Get the underlying Ray actor handle, resolving by name if needed."""
        if self._actor_ref is None:
            object.__setattr__(self, "_actor_ref", ray.get_actor(self._actor_name))
        return self._actor_ref

    def __getattr__(self, method_name: str) -> RayProxyMethod:
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        return RayProxyMethod(self._resolve(), method_name)

    def __getstate__(self) -> dict:
        """Serialize by name for cross-process transfer."""
        if self._actor_name:
            return {"name": self._actor_name}
        return {"ref": self._actor_ref}

    def __setstate__(self, state: dict) -> None:
        """Deserialize from name or ref."""
        if "name" in state:
            object.__setattr__(self, "_actor_name", state["name"])
            object.__setattr__(self, "_actor_ref", None)
        else:
            object.__setattr__(self, "_actor_name", None)
            object.__setattr__(self, "_actor_ref", state["ref"])


class RayProxyMethod:
    """Wraps a method call that goes through _RayActorHost._proxy_call."""

    def __init__(self, ray_handle: ray.actor.ActorHandle, method_name: str):
        self._ray_handle = ray_handle
        self._method_name = method_name

    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        object_ref = self._ray_handle._proxy_call.remote(self._method_name, args, kwargs)
        return RayActorFuture(object_ref)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        object_ref = self._ray_handle._proxy_call.remote(self._method_name, args, kwargs)
        return ray.get(object_ref)


class RayActorFuture:
    """ActorFuture backed by a Ray ObjectRef."""

    def __init__(self, object_ref: ray.ObjectRef):
        self._object_ref = object_ref

    def result(self, timeout: float | None = None) -> Any:
        return ray.get(self._object_ref, timeout=timeout)


class RayActorGroup:
    """ActorGroup for Ray actors. All actors are ready immediately after creation."""

    def __init__(self, handles: list[ActorHandle]):
        self._handles = handles
        self._yielded = False

    @property
    def ready_count(self) -> int:
        """All Ray actors are ready immediately after creation."""
        return len(self._handles)

    def wait_ready(self, count: int | None = None, timeout: float = 300.0) -> list[ActorHandle]:
        """Return ready actor handles. Ray actors are ready immediately."""
        if count is None:
            count = len(self._handles)
        self._yielded = True
        return self._handles[:count]

    def discover_new(self) -> list[ActorHandle]:
        """Return handles not yet yielded. After wait_ready, returns empty."""
        if self._yielded:
            return []
        self._yielded = True
        return self._handles

    def shutdown(self) -> None:
        """Kill all Ray actors."""
        for handle in self._handles:
            try:
                ray.kill(handle._actor_ref)
            except Exception as e:
                logger.warning("Failed to kill Ray actor: %s", e)
