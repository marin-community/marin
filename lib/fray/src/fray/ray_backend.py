# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RayClient: fray backend that runs jobs as Ray tasks and actors as named Ray actors.

Jobs are single Ray tasks that return their outcome (error, formatted remote
traceback and a bounded log tail) instead of raising, so the original exception
type crosses the wire next to the logs. Each fray actor is one generic
``_ActorHost`` Ray actor named ``{name}-{index}`` in the client's private
namespace. Actors are owned by the process that created them, so Ray reaps a
driver's actors when the driver dies and a coordinator's worker group when the
coordinator dies, matching Iris's cascading termination. Handles pickle to
``namespace/name-index`` and re-resolve through ``ray.get_actor`` in any
process of the cluster. Inside tasks and actors ``current_client()`` finds this
backend through ``FRAY_BACKEND=ray``: ``connect`` records that marker plus any
``runtime_env`` the caller passes (for example ``py_executable``) on the client,
and every task and actor the client creates carries it in its own Ray runtime
env, so workers started under a pinned interpreter see the same backend.

Import this module directly (``from fray.ray_backend import RayClient``); core
fray modules never import it, so ``ray`` stays an optional dependency.
"""

import collections
import contextlib
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, TextIO

import humanfriendly
import ray
import ray.actor
import ray.util
from ray._private.ray_constants import RAY_ENABLE_UV_RUN_RUNTIME_ENV
from ray._private.runtime_env.uv_runtime_env_hook import _get_uv_run_cmdline
from ray.exceptions import ActorAlreadyExistsError, RayActorError, RayError, RayTaskError
from ray.util.accelerators import accelerators as ray_accelerators

from fray.actor import (
    ActorContext,
    ActorFuture,
    ActorGroup,
    ActorHandle,
    ActorUnavailableError,
    HostedActor,
    _reset_current_actor,
    _set_current_actor,
)
from fray.current_client import BACKEND_ENV, RAY_BACKEND, set_current_client
from fray.local_backend import _run_binary, _run_callable
from fray.types import (
    ActorConfig,
    Entrypoint,
    EnvironmentConfig,
    GpuConfig,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
)

logger = logging.getLogger(__name__)

LOG_TAIL_LINES = 10_000
HOSTED_ENDPOINT_PREFIX = "hosted/"
READY_POLL_INTERVAL = 0.5
ACTOR_INIT_RETRY_DELAY = 1.0
GPU_VARIANT_ANY = "auto"

# Ray's accelerator_type names (``H100``, ``A100-80G``, ``L4``, ...); fray GPU variants map onto them.
RAY_ACCELERATOR_TYPES: frozenset[str] = frozenset(
    getattr(ray_accelerators, name) for name in dir(ray_accelerators) if name.isupper()
)

# In-process actors served by host_actor(); keyed by endpoint, resolvable only in this process.
_hosted_registry: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Resource mapping
# ---------------------------------------------------------------------------


def ray_options(resources: ResourceConfig, actor_config: ActorConfig | None = None) -> dict[str, Any]:
    """Translate fray resources into ``.options(...)`` kwargs for a Ray task or actor.

    Actors request ``num_cpus=0`` so they never contend for head-node CPU
    slots; tasks request ``resources.cpu``. ``ram`` becomes Ray's ``memory``
    resource in bytes. A GPU variant other than ``auto`` becomes Ray's
    ``accelerator_type`` so the request lands on that GPU model only.
    Placement fields (``regions``, ``zone``, ``target_cluster``), TPU devices
    and GPU variants Ray has no accelerator name for are rejected; ``disk``,
    ``preemptible``, ``image`` and ``device_alternatives`` have no Ray
    equivalent and are ignored.

    Args:
        resources: Per-replica resource request.
        actor_config: Present for actors; maps concurrency and restart policy.
            ``max_task_retries`` follows the Iris meaning of "restart the
            replica", so it feeds Ray's ``max_restarts``.

    Raises:
        ValueError: For TPU devices, unknown GPU variants or placement constraints.
    """
    if isinstance(resources.device, TpuConfig):
        raise ValueError("TPU devices cannot be scheduled on the Ray backend")
    if resources.regions is not None or resources.zone is not None or resources.target_cluster is not None:
        raise ValueError(
            "regions/zone/target_cluster are not schedulable on the Ray backend: "
            f"regions={resources.regions} zone={resources.zone} target_cluster={resources.target_cluster}"
        )
    logger.debug(
        "Ray backend ignores disk=%s preemptible=%s image=%s device_alternatives=%s",
        resources.disk,
        resources.preemptible,
        resources.image,
        resources.device_alternatives,
    )

    options: dict[str, Any] = {
        "num_cpus": 0 if actor_config is not None else resources.cpu,
        "memory": humanfriendly.parse_size(resources.ram, binary=True),
    }
    if isinstance(resources.device, GpuConfig):
        options["num_gpus"] = resources.device.count
        if resources.device.variant != GPU_VARIANT_ANY:
            options["accelerator_type"] = _ray_accelerator_type(resources.device.variant)
    if actor_config is not None:
        options["max_concurrency"] = actor_config.max_concurrency
        options["max_restarts"] = actor_config.max_task_retries or actor_config.max_restarts or 0
        options["max_task_retries"] = 0
    return options


def _ray_accelerator_type(variant: str) -> str:
    accelerator = variant.upper()
    if accelerator not in RAY_ACCELERATOR_TYPES:
        raise ValueError(f"GPU variant {variant!r} has no Ray accelerator_type; known: {sorted(RAY_ACCELERATOR_TYPES)}")
    return accelerator


def ray_runtime_env(environment: EnvironmentConfig | None) -> dict[str, Any] | None:
    """Translate an EnvironmentConfig into a Ray ``runtime_env`` dict, or None when unset.

    A workspace environment pins workers to the driver's interpreter; a docker
    image environment runs workers in that container. ``pip_packages`` and
    ``setup_scripts`` are rejected: Ray's pip plugin clones the driver's venv
    and needs ``pip`` inside it, which uv-managed venvs lack, and Ray runs no
    setup scripts. ``extras`` and ``sync_packages`` only shape the uv sync
    that the Ray backend never performs and are ignored.

    Raises:
        ValueError: For ``pip_packages`` or a non-empty ``setup_scripts`` list.
    """
    if environment is None:
        return None
    if environment.pip_packages:
        raise ValueError(f"pip_packages cannot be installed on the Ray backend: {list(environment.pip_packages)}")
    if environment.setup_scripts:
        raise ValueError(f"setup_scripts do not run on the Ray backend: {list(environment.setup_scripts)}")
    runtime_env: dict[str, Any] = {}
    if environment.workspace:
        runtime_env["py_executable"] = sys.executable
    else:
        runtime_env["container"] = {"image": environment.docker_image}
    if environment.env_vars:
        runtime_env["env_vars"] = dict(environment.env_vars)
    if environment.extras or environment.sync_packages:
        logger.debug(
            "Ray backend ignores workspace extras=%s sync_packages=%s", environment.extras, environment.sync_packages
        )
    return runtime_env


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class RemoteTraceback(Exception):
    """The formatted traceback of an exception raised inside a Ray job; attached as ``__cause__``."""


@dataclass(frozen=True)
class _JobOutcome:
    """What a job task returns: the exception it raised (if any), its formatted traceback and log tail."""

    error: BaseException | None
    traceback: str | None
    log_lines: tuple[str, ...]


class _LineSink:
    """Collects written text into a bounded deque, one entry per completed line.

    Carriage returns count as line ends so progress bars that only ever emit
    ``\\r`` land in the tail instead of growing the pending partial line.
    """

    def __init__(self, lines: collections.deque[str]):
        self._lines = lines
        self._partial = ""

    def write(self, text: str) -> int:
        self._partial += text.replace("\r\n", "\n").replace("\r", "\n")
        *complete, self._partial = self._partial.split("\n")
        self._lines.extend(complete)
        return len(text)

    def flush(self) -> None:
        pass

    def flush_partial(self) -> None:
        """Record the unterminated last line, if any."""
        if self._partial:
            self._lines.append(self._partial)
            self._partial = ""


class _TeeStream(_LineSink):
    """A stdout/stderr replacement that records lines and forwards writes to the original stream."""

    def __init__(self, stream: TextIO, lines: collections.deque[str]):
        super().__init__(lines)
        self._stream = stream

    def write(self, text: str) -> int:
        super().write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _LogCapture:
    """Retain a bounded tail of root-logger records plus stdout/stderr writes for one job."""

    def __init__(self, max_lines: int):
        self.lines: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._sinks: list[_LineSink] = [_LineSink(self.lines)]
        self._handler = logging.StreamHandler(self._sinks[0])
        self._handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self._redirects = contextlib.ExitStack()
        self._root_level = logging.NOTSET

    def __enter__(self) -> "_LogCapture":
        root = logging.getLogger()
        self._root_level = root.level
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(self._handler)
        for redirect, stream in ((contextlib.redirect_stdout, sys.stdout), (contextlib.redirect_stderr, sys.stderr)):
            tee = _TeeStream(stream, self.lines)
            self._sinks.append(tee)
            self._redirects.enter_context(redirect(tee))
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._redirects.close()
        root = logging.getLogger()
        root.removeHandler(self._handler)
        root.setLevel(self._root_level)
        for sink in self._sinks:
            sink.flush_partial()

    def tail(self) -> tuple[str, ...]:
        return tuple(self.lines)


def _run_entrypoint(entrypoint: Entrypoint) -> None:
    if entrypoint.callable_entrypoint is not None:
        _run_callable(entrypoint.callable_entrypoint)
        return
    # submit() rejects an entrypoint with neither form before it reaches a worker.
    assert entrypoint.binary_entrypoint is not None
    _run_binary(entrypoint.binary_entrypoint)


# max_retries is set per submission from JobRequest.max_retries_failure; the decorator default
# of 0 keeps a bare call from inheriting Ray's three system-failure retries.
@ray.remote(max_retries=0)
def _run_job(entrypoint: Entrypoint, max_log_lines: int) -> _JobOutcome:
    """Run one job inside a Ray worker, returning its error and log tail instead of raising."""
    os.environ[BACKEND_ENV] = RAY_BACKEND
    capture = _LogCapture(max_log_lines)
    with capture, set_current_client(RayClient.attach()):
        try:
            _run_entrypoint(entrypoint)
        except Exception as exc:
            formatted = "".join(traceback.format_exception(exc))
            return _JobOutcome(error=exc, traceback=formatted, log_lines=capture.tail())
    return _JobOutcome(error=None, traceback=None, log_lines=capture.tail())


class RayJobHandle:
    """Job handle backed by the ObjectRef of a ``_run_job`` task.

    A ``JobRequest.timeout`` is enforced by a timer that cancels the task at
    the deadline whether or not anyone polls the handle. Once the outcome is
    fetched, or the job is terminated, the handle answers from its own state
    and no longer touches the Ray runtime.
    """

    def __init__(self, job_id: str, ref: ray.ObjectRef, timeout: float | None):
        self._job_id = job_id
        self._ref = ref
        self._deadline = None if timeout is None else time.monotonic() + timeout
        self._terminated = threading.Event()
        self._outcome: _JobOutcome | None = None
        self._deadline_timer: threading.Timer | None = None
        if timeout is not None:
            self._deadline_timer = threading.Timer(timeout, self._expire)
            self._deadline_timer.daemon = True
            self._deadline_timer.start()

    @property
    def job_id(self) -> str:
        return self._job_id

    def _expire(self) -> None:
        ready, _ = ray.wait([self._ref], timeout=0)
        if not ready:
            self.terminate()

    def _stop_deadline_timer(self) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()

    def _fetch(self, timeout: float | None) -> _JobOutcome:
        """Block up to ``timeout`` for the task's outcome; Ray-level failures become outcomes too.

        Raises:
            ray.exceptions.GetTimeoutError: When the task has not finished in time.
        """
        if self._outcome is not None:
            return self._outcome
        try:
            outcome = ray.get(self._ref, timeout=timeout)
        except TimeoutError:
            raise
        except RayError as exc:
            outcome = _JobOutcome(error=exc, traceback=None, log_lines=())
        self._outcome = outcome
        self._stop_deadline_timer()
        return outcome

    def _remaining(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def _finished_status(self) -> JobStatus:
        assert self._outcome is not None
        if self._outcome.error is not None:
            return JobStatus.FAILED
        return JobStatus.SUCCEEDED

    def status(self) -> JobStatus:
        if self._terminated.is_set():
            return JobStatus.STOPPED
        if self._outcome is not None:
            return self._finished_status()
        ready, _ = ray.wait([self._ref], timeout=0)
        if not ready:
            if self._remaining() == 0.0:
                self.terminate()
                return JobStatus.STOPPED
            return JobStatus.RUNNING
        self._fetch(timeout=None)
        return self._finished_status()

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        """Block until the job completes, its own deadline passes, or ``timeout`` expires.

        A job deadline (``JobRequest.timeout``) or an earlier ``terminate()``
        yields STOPPED. A caller ``timeout`` propagates as
        ``ray.exceptions.GetTimeoutError``, a ``TimeoutError`` subclass. With
        ``raise_on_failure`` the job's own exception is re-raised with the
        remote traceback attached as a ``RemoteTraceback`` cause.
        """
        if self._terminated.is_set():
            return JobStatus.STOPPED
        remaining = self._remaining()
        deadline_binding = remaining is not None and (timeout is None or remaining <= timeout)
        try:
            outcome = self._fetch(timeout=remaining if deadline_binding else timeout)
        except TimeoutError:
            if not deadline_binding:
                raise
            self.terminate()
            return JobStatus.STOPPED
        if self._terminated.is_set():
            return JobStatus.STOPPED
        if raise_on_failure and outcome.error is not None:
            if outcome.traceback is not None:
                outcome.error.__cause__ = RemoteTraceback(outcome.traceback)
            raise outcome.error
        return self._finished_status()

    def logs(self, max_lines: int = 0) -> tuple[str, ...]:
        """Return the tail of the job's captured log lines; empty until the job finishes."""
        if self._outcome is None:
            ready, _ = ray.wait([self._ref], timeout=0)
            if not ready:
                return ()
            self._fetch(timeout=None)
        lines = self._outcome.log_lines
        if max_lines > 0:
            return lines[-max_lines:]
        return lines

    def terminate(self) -> None:
        self._terminated.set()
        self._stop_deadline_timer()
        ray.cancel(self._ref, force=True)


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------


class _ActorHost:
    """Generic Ray actor that hosts one fray actor instance and serves its methods.

    The constructor is attempted ``max_init_attempts`` times; a final failure
    is reported through ``ready()`` and the group kills the member.
    """

    def __init__(
        self,
        endpoint: str,
        index: int,
        group_name: str,
        actor_class: type,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        max_init_attempts: int,
    ):
        os.environ[BACKEND_ENV] = RAY_BACKEND
        handle = RayActorHandle(endpoint)
        self._ctx = ActorContext(handle=handle, index=index, group_name=group_name, shutdown_event=threading.Event())
        self._init_error: BaseException | None = None
        self._instance: Any = None
        token = _set_current_actor(self._ctx)
        try:
            self._instance = self._construct(actor_class, args, kwargs, max_init_attempts)
        except BaseException as exc:
            # Kept, not raised: raising from a Ray __init__ surfaces only as an
            # ActorDiedError without the cause, while ready() can re-raise the exact class.
            self._init_error = exc
            return
        finally:
            _reset_current_actor(token)
        threading.Thread(target=self._watch_shutdown, name=f"fray-actor-shutdown-{endpoint}", daemon=True).start()

    def _construct(self, actor_class: type, args: tuple[Any, ...], kwargs: dict[str, Any], attempts: int) -> Any:
        attempt = 1
        while True:
            try:
                return actor_class(*args, **kwargs)
            except Exception as exc:
                if attempt >= attempts:
                    raise
                logger.warning(
                    "Actor %s init attempt %d/%d failed; retrying: %s", self._ctx.handle, attempt, attempts, exc
                )
                attempt += 1
                time.sleep(ACTOR_INIT_RETRY_DELAY)

    def ready(self) -> str:
        """Re-raise a constructor failure, else return the id of the Ray node hosting this actor."""
        if self._init_error is not None:
            raise self._init_error
        return ray.get_runtime_context().get_node_id()

    def call(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if self._init_error is not None:
            raise self._init_error
        method = getattr(self._instance, name)
        if not callable(method):
            raise AttributeError(f"{name} is not callable on {type(self._instance).__name__}")
        return method(*args, **kwargs)

    def _watch_shutdown(self) -> None:
        assert self._ctx.shutdown_event is not None
        self._ctx.shutdown_event.wait()
        if self._ctx._errors:
            logger.error("Actor %s failed: %s", self._ctx.handle, self._ctx._errors[0], exc_info=self._ctx._errors[0])
            # A dirty exit is what makes Ray apply max_restarts; sys.exit is a no-op off the main thread.
            os._exit(1)
        ray.actor.exit_actor()


_RemoteActorHost = ray.remote(_ActorHost)


def _thread_future(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Future[Any]:
    """Run ``fn`` on a dedicated daemon thread (never pooled: pooled callbacks deadlock coordinators)."""
    future: Future[Any] = Future()

    def run() -> None:
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return future


class RayActorHandle:
    """Handle that pickles to an endpoint and re-resolves the Ray actor lazily.

    Endpoints are ``namespace/name-index`` for Ray-hosted actors and
    ``hosted/name-0`` for actors served in-process by ``host_actor``; the
    latter resolve only inside the hosting process.
    """

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._actor: ray.actor.ActorHandle | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"RayActorHandle({self._endpoint!r})"

    def _resolve(self) -> Any:
        """Return the Ray actor handle for this endpoint, cached until an actor error."""
        with self._lock:
            if self._actor is None:
                namespace, _, name = self._endpoint.partition("/")
                try:
                    self._actor = ray.get_actor(name, namespace=namespace)
                except ValueError as exc:
                    raise ActorUnavailableError(f"Actor {self._endpoint} not found") from exc
            return self._actor

    def _invalidate(self) -> None:
        with self._lock:
            self._actor = None

    def _invoke(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ActorFuture:
        if self._endpoint.startswith(HOSTED_ENDPOINT_PREFIX):
            instance = _hosted_registry.get(self._endpoint)
            if instance is None:
                raise ActorUnavailableError(f"Hosted actor {self._endpoint} is not served in this process")
            return _thread_future(getattr(instance, name), args, kwargs)
        actor = self._resolve()
        try:
            ref = actor.call.remote(name, args, kwargs)
        except RayActorError as exc:
            self._invalidate()
            raise ActorUnavailableError(f"Actor {self._endpoint} unavailable: {exc}") from exc
        return _RayFuture(ref, self)

    def __getattr__(self, method_name: str) -> "_RayActorMethod":
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        return _RayActorMethod(self, method_name)

    def __getstate__(self) -> dict[str, str]:
        return {"endpoint": self._endpoint}

    def __setstate__(self, state: dict[str, str]) -> None:
        self._endpoint = state["endpoint"]
        self._actor = None
        self._lock = threading.Lock()


class _RayActorMethod:
    """One method on a RayActorHandle; every call is an RPC through ``_ActorHost.call``."""

    def __init__(self, handle: RayActorHandle, name: str):
        self._handle = handle
        self._name = name

    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
        return self._handle._invoke(self._name, args, kwargs)

    def submit(self, *args: Any, **kwargs: Any) -> ActorFuture:
        """Long-running variant; Ray actor calls already survive without a held connection."""
        return self.remote(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.remote(*args, **kwargs).result()


class _RayFuture:
    """ActorFuture over an ObjectRef that unwraps remote errors to their original type."""

    def __init__(self, ref: ray.ObjectRef, handle: RayActorHandle):
        self._ref = ref
        self._handle = handle

    def result(self, timeout: float | None = None) -> Any:
        """Block for the result; a timeout leaves the call running and the future re-pollable.

        Raises:
            The actor method's own exception, with its original class.
            ActorUnavailableError: The actor died, restarted, or is unreachable.
            ray.exceptions.GetTimeoutError: ``timeout`` expired (a ``TimeoutError`` subclass).
        """
        try:
            return ray.get(self._ref, timeout=timeout)
        except RayTaskError as err:
            if err.cause is None:
                raise
            raise err.cause from err
        except RayActorError as exc:
            self._handle._invalidate()
            raise ActorUnavailableError(f"Actor {self._handle} unavailable: {exc}") from exc


class RayActorGroup:
    """Group of ``count`` ``_ActorHost`` actors named ``{name}-{i}`` in one namespace.

    A member whose constructor fails (after its retries) is killed and counted
    dead when its readiness probe reports; the other members keep running, as
    on Iris. ``wait_ready`` raises that constructor error only once the
    requested count can no longer be reached.

    Pickles to ``(name, namespace, count)``; a receiving process re-resolves
    members by name.
    """

    def __init__(self, name: str, namespace: str, count: int):
        self.name = name
        self.namespace = namespace
        self.count = count
        self.handles = [RayActorHandle(f"{namespace}/{name}-{i}") for i in range(count)]
        self._ready: set[int] = set()
        self._yielded: set[int] = set()
        self._dead: set[int] = set()
        self._failures: dict[int, BaseException] = {}
        self._probes: dict[ray.ObjectRef, int] = {}

    def __getstate__(self) -> dict[str, Any]:
        return {"name": self.name, "namespace": self.namespace, "count": self.count}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["name"], state["namespace"], state["count"])  # type: ignore[misc]

    def _member_name(self, index: int) -> str:
        return f"{self.name}-{index}"

    def _issue_probes(self) -> None:
        probed = set(self._probes.values())
        for index in range(self.count):
            if index in self._ready or index in self._dead or index in probed:
                continue
            try:
                actor = self.handles[index]._resolve()
            except ActorUnavailableError:
                self._dead.add(index)
                continue
            self._probes[actor.ready.remote()] = index

    def _kill_member(self, index: int, error: BaseException) -> None:
        with contextlib.suppress(ActorUnavailableError):
            ray.kill(self.handles[index]._resolve(), no_restart=True)
        self.handles[index]._invalidate()
        self._dead.add(index)
        self._failures[index] = error
        logger.error("Actor %s/%s failed to start: %s", self.namespace, self._member_name(index), error)

    def _collect_ready(self, timeout: float) -> None:
        """Issue readiness probes and absorb those that finish within ``timeout``.

        A probe that raises the constructor's error, or whose actor died, kills
        that member and records the error; a probe against a restarting actor
        is simply re-issued.
        """
        self._issue_probes()
        if not self._probes:
            return
        finished, _ = ray.wait(list(self._probes), num_returns=len(self._probes), timeout=timeout)
        for ref in finished:
            index = self._probes.pop(ref)
            try:
                node_id = ray.get(ref)
            except RayTaskError as err:
                self._kill_member(index, err.cause if err.cause is not None else err)
                continue
            except ray.exceptions.ActorUnavailableError:
                self.handles[index]._invalidate()
                continue
            except RayActorError as exc:
                died = RuntimeError(f"Actor {self._member_name(index)} died before becoming ready")
                died.__cause__ = exc
                self._kill_member(index, died)
                continue
            logger.info("Actor %s/%s ready on Ray node %s", self.namespace, self._member_name(index), node_id)
            self._ready.add(index)

    def _take(self, indices: Iterable[int]) -> list[ActorHandle]:
        ordered = sorted(indices)
        self._yielded.update(ordered)
        return [self.handles[i] for i in ordered]

    @property
    def ready_count(self) -> int:
        self._collect_ready(timeout=0)
        return len(self._ready)

    def wait_ready(self, count: int | None = None, timeout: float = 300.0) -> list[ActorHandle]:
        """Block until ``count`` members (default: all) answered a readiness probe.

        Returns the ready handles in index order. Actors whose resources can
        never be satisfied stay pending silently, so this raises
        ``TimeoutError`` at the deadline rather than spinning.

        Raises:
            TimeoutError: ``count`` members were not ready within ``timeout``.
            RuntimeError: Enough members died that ``count`` is unreachable; when
                a member's constructor failed, that exception is raised instead.
        """
        if count is None:
            count = self.count
        deadline = time.monotonic() + timeout
        while len(self._ready) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Only {len(self._ready)}/{count} actors of {self.name} ready after {timeout}s")
            self._collect_ready(timeout=min(READY_POLL_INTERVAL, remaining))
            if self.count - len(self._dead) < count:
                raise self._unreachable_error(count)
        return self._take(sorted(self._ready)[:count])

    def _unreachable_error(self, count: int) -> BaseException:
        if self._failures:
            return self._failures[min(self._failures)]
        return RuntimeError(
            f"Actor group {self.name} can no longer reach {count} ready members: "
            f"{len(self._ready)} ready, {len(self._dead)} dead"
        )

    def discover_new(self) -> list[ActorHandle]:
        self._collect_ready(timeout=0)
        return self._take(self._ready - self._yielded)

    def is_done(self) -> bool:
        """True once no member is registered any more (exhausted restarts, clean exit, killed, or failed init)."""
        self._collect_ready(timeout=0)
        if len(self._dead) == self.count:
            return True
        # The owner's pinned creation handle keeps ray.get_actor resolving a dead actor;
        # the named-actor registry drops it as soon as it dies.
        registered = {
            entry["name"]
            for entry in ray.util.list_named_actors(all_namespaces=True)
            if entry["namespace"] == self.namespace
        }
        for index in range(self.count):
            if index in self._dead:
                continue
            if self._member_name(index) in registered:
                return False
            self._dead.add(index)
        return True

    def shutdown(self) -> None:
        for index, handle in enumerate(self.handles):
            with contextlib.suppress(ActorUnavailableError):
                ray.kill(handle._resolve(), no_restart=True)
            handle._invalidate()
            self._dead.add(index)
        self._ready.clear()
        self._probes.clear()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RayClient:
    """Fray Client on a Ray cluster.

    Create one with ``connect`` from a driver, or ``attach`` from inside a Ray
    task or actor where ``ray`` is already initialized. The client that
    initializes the runtime mints a private actor namespace so actor names
    never collide across drivers; ``shutdown`` kills everything created in
    that namespace.
    """

    _attached: "RayClient | None" = None
    _attach_lock = threading.Lock()

    # Applied to every actor and task this backend creates. A job-level runtime env set
    # at ``ray.init`` does not reach workers started for actors under a pinned
    # interpreter, and nested drivers must forward the backend marker themselves.
    def __init__(self, namespace: str, owns_runtime: bool, worker_runtime_env: dict[str, Any] | None = None):
        self.namespace = namespace
        self._owns_runtime = owns_runtime
        # Attached to every task and actor this client creates. Workers that attach()
        # add only the backend marker; Ray hands them the job-level env (for example
        # py_executable) that connect() recorded on the driver.
        self._worker_runtime_env = worker_runtime_env or {"env_vars": {BACKEND_ENV: RAY_BACKEND}}
        self._jobs: list[RayJobHandle] = []
        # Creation handles by endpoint. Ray reaps an owned actor once its creation handle is
        # collected, so the client pins them until kill_actors(): an actor lives for the
        # client's lifetime, not for the lifetime of whichever handle the caller kept.
        self._owned_actors: dict[str, ray.actor.ActorHandle] = {}
        self._closed = False

    @classmethod
    def connect(cls, address: str = "auto", namespace: str | None = None, **init_kwargs: Any) -> "RayClient":
        """Initialize Ray and return a client; the client owns the runtime only if it initialized it.

        When Ray is already initialized in this process the client joins the
        runtime's namespace; ``shutdown`` then leaves the runtime up for the
        client that started it.

        Args:
            address: Passed to ``ray.init``; ``"auto"`` joins a running cluster,
                ``"local"`` starts a fresh single-node one.
            namespace: Actor namespace; defaults to a fresh ``fray-<uuid>``.
            **init_kwargs: Extra ``ray.init`` keyword arguments (``num_cpus``, ...).

        Raises:
            RuntimeError: The driver runs under ``uv run`` with Ray's uv hook
                enabled, which would launch workers without ``ray`` installed and
                hang the first ``ray.get``. Export ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0``
                before starting the driver.
            ValueError: Ray is already initialized and ``namespace`` or
                ``init_kwargs`` ask for something the running runtime cannot honour.
        """
        if RAY_ENABLE_UV_RUN_RUNTIME_ENV and _get_uv_run_cmdline():
            raise RuntimeError(
                "Ray's uv runtime-env hook is active for this `uv run` driver; workers would start "
                "without ray installed. Export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 before importing ray."
            )
        if ray.is_initialized():
            runtime_namespace = ray.get_runtime_context().namespace
            if namespace is not None and namespace != runtime_namespace:
                raise ValueError(
                    f"Ray is already initialized in namespace {runtime_namespace!r}; "
                    f"cannot connect with namespace {namespace!r}"
                )
            if init_kwargs:
                raise ValueError(f"Ray is already initialized; init kwargs cannot apply: {sorted(init_kwargs)}")
            client = cls(runtime_namespace, owns_runtime=False)
        else:
            # Callers may pin the worker interpreter (``py_executable``) or extra env vars;
            # the backend marker is always added so workers resolve this client.
            runtime_env = dict(init_kwargs.pop("runtime_env", None) or {})
            runtime_env["env_vars"] = {**runtime_env.get("env_vars", {}), BACKEND_ENV: RAY_BACKEND}
            ray.init(
                address=address,
                namespace=namespace or f"fray-{uuid.uuid4().hex[:8]}",
                include_dashboard=False,
                runtime_env=runtime_env,
                **init_kwargs,
            )
            client = cls(ray.get_runtime_context().namespace, owns_runtime=True, worker_runtime_env=runtime_env)
        os.environ[BACKEND_ENV] = RAY_BACKEND
        with cls._attach_lock:
            if cls._attached is None:
                cls._attached = client
        return client

    @classmethod
    def attach(cls) -> "RayClient":
        """Return the process-wide client for the already-initialized Ray runtime.

        Inside tasks and actors the runtime namespace equals the driver's, so
        handles created here resolve the same actors.

        Raises:
            RuntimeError: Ray is not initialized in this process.
        """
        with cls._attach_lock:
            if not ray.is_initialized():
                raise RuntimeError("RayClient.attach() requires an initialized Ray runtime; use RayClient.connect()")
            namespace = ray.get_runtime_context().namespace
            if cls._attached is None or cls._attached.namespace != namespace:
                cls._attached = cls(namespace, owns_runtime=False)
            return cls._attached

    def submit(self, request: JobRequest, adopt_existing: bool = True) -> RayJobHandle:
        """Run the request as one Ray task and return its handle immediately.

        ``max_retries_failure`` bounds Ray's retries of a task whose worker
        died (crash, OOM, lost node); the job's own exceptions are returned as
        outcomes and never retried. Ray cannot tell a preemption from a crash,
        so ``max_retries_preemption`` and ``max_task_failures`` are ignored,
        as is ``priority``. Job names are not enforced to be unique (as in
        LocalClient), so ``adopt_existing`` has no effect.

        Raises:
            ValueError: The entrypoint is empty, ``replicas`` > 1, or
                ``processes_per_task`` > 1 (Ray has no multi-process supervisor).
        """
        del adopt_existing
        entrypoint = request.entrypoint
        if entrypoint.callable_entrypoint is None and entrypoint.binary_entrypoint is None:
            raise ValueError("JobRequest entrypoint must have either callable_entrypoint or binary_entrypoint")
        if request.replicas not in (None, 1):
            raise ValueError(f"Ray backend runs single-replica jobs only, got replicas={request.replicas}")
        if request.processes_per_task != 1:
            raise ValueError(
                f"processes_per_task={request.processes_per_task} is not supported on the Ray backend "
                "(one process per task)"
            )
        logger.debug(
            "Ray backend ignores max_retries_preemption=%s max_task_failures=%s priority=%s",
            request.max_retries_preemption,
            request.max_task_failures,
            request.priority,
        )
        options: dict[str, Any] = {
            "name": f"fray-job/{request.name}",
            "max_retries": request.max_retries_failure,
            **ray_options(request.resources),
        }
        runtime_env = {**self._worker_runtime_env, **(ray_runtime_env(request.environment) or {})}
        runtime_env["env_vars"] = {
            **self._worker_runtime_env.get("env_vars", {}),
            **runtime_env.get("env_vars", {}),
        }
        options["runtime_env"] = runtime_env
        ref = _run_job.options(**options).remote(entrypoint, LOG_TAIL_LINES)
        job_id = f"ray-{request.name}-{uuid.uuid4().hex[:8]}"
        timeout = request.timeout.to_seconds() if request.timeout is not None else None
        handle = RayJobHandle(job_id, ref, timeout)
        self._jobs.append(handle)
        return handle

    def host_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        actor_config: ActorConfig = ActorConfig(),
        **kwargs: Any,
    ) -> HostedActor:
        """Host an actor in this process; its handle resolves only here, not from Ray workers."""
        del actor_config
        endpoint = f"{HOSTED_ENDPOINT_PREFIX}{name}-0"
        handle = RayActorHandle(endpoint)
        shutdown_event = threading.Event()
        ctx = ActorContext(handle=handle, index=0, group_name=name, shutdown_event=shutdown_event)
        token = _set_current_actor(ctx)
        try:
            instance = actor_class(*args, **kwargs)
        finally:
            _reset_current_actor(token)
        _hosted_registry[endpoint] = instance

        def stop() -> None:
            _hosted_registry.pop(endpoint, None)
            shutdown_event.set()

        def stop_when_actor_asks() -> None:
            shutdown_event.wait()
            stop()

        threading.Thread(target=stop_when_actor_asks, daemon=True).start()
        return HostedActor(handle, stop=stop)

    def create_actor(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        resources: ResourceConfig = ResourceConfig(),
        actor_config: ActorConfig = ActorConfig(),
        **kwargs: Any,
    ) -> ActorHandle:
        """Create one actor and block until its constructor has run; a failed create leaves nothing behind."""
        group = self.create_actor_group(
            actor_class, *args, name=name, count=1, resources=resources, actor_config=actor_config, **kwargs
        )
        try:
            return group.wait_ready()[0]
        except BaseException:
            group.shutdown()
            raise

    def create_actor_group(
        self,
        actor_class: type,
        *args: Any,
        name: str,
        count: int,
        resources: ResourceConfig = ResourceConfig(),
        actor_config: ActorConfig = ActorConfig(),
        **kwargs: Any,
    ) -> ActorGroup:
        """Create ``count`` actors owned by this process without waiting for them; see ``RayActorGroup.wait_ready``.

        The actors live until the group or client kills them, or this process
        dies; dropping the returned group or handles does not end them.

        Raises:
            ValueError: A member name is already taken in this namespace, which
                means an earlier group with this name in the namespace was not shut down.
        """
        options = {
            "namespace": self.namespace,
            "runtime_env": self._worker_runtime_env,
            # Ray packs one-CPU actors onto the first node with room; a multi-node
            # cluster only helps if the group is spread across its nodes.
            "scheduling_strategy": "SPREAD",
            **ray_options(resources, actor_config),
        }
        max_init_attempts = 1 + (actor_config.max_task_retries or 0)
        group = RayActorGroup(name, self.namespace, count)
        created: list[ray.actor.ActorHandle] = []
        try:
            for index in range(count):
                member = group._member_name(index)
                try:
                    # ray.remote() is typed as a union with the class itself; at runtime it is a handle.
                    actor: ray.actor.ActorHandle = _RemoteActorHost.options(name=member, **options).remote(  # type: ignore[assignment]
                        f"{self.namespace}/{member}", index, name, actor_class, args, kwargs, max_init_attempts
                    )
                except ActorAlreadyExistsError as exc:
                    raise ValueError(
                        f"Actor {member} already exists in namespace {self.namespace}; "
                        "an earlier group with this name was not shut down"
                    ) from exc
                created.append(actor)
                group.handles[index]._actor = actor
        except BaseException:
            for actor in created:
                ray.kill(actor, no_restart=True)
            raise
        for handle, actor in zip(group.handles, created, strict=True):
            self._owned_actors[handle._endpoint] = actor
        return group

    def kill_actors(self) -> None:
        """Kill every named actor in this client's namespace, restarts included."""
        for entry in ray.util.list_named_actors(all_namespaces=True):
            if entry["namespace"] != self.namespace:
                continue
            with contextlib.suppress(ValueError):
                ray.kill(ray.get_actor(entry["name"], namespace=self.namespace), no_restart=True)
        self._owned_actors.clear()

    def shutdown(self, wait: bool = True) -> None:
        """Release everything this client created; later calls are no-ops.

        With ``wait`` the client blocks for running jobs first (each up to its
        own deadline); otherwise it terminates them. Every actor in the
        client's namespace is killed, and the Ray runtime is shut down only if
        this client initialized it. Job handles keep answering from their
        final state afterwards.
        """
        if self._closed:
            return
        self._closed = True
        for job in self._jobs:
            if wait:
                job.wait(raise_on_failure=False)
            elif not JobStatus.finished(job.status()):
                job.terminate()
        self.kill_actors()
        with self._attach_lock:
            if RayClient._attached is self:
                RayClient._attached = None
        if self._owns_runtime:
            os.environ.pop(BACKEND_ENV, None)
            ray.shutdown()
