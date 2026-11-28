# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate an Iris task gang into vLLM's native multiprocess arguments."""

import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.actor.client import ActorClient
from iris.actor.server import ActorServer
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.rpc.errors import is_retryable_error
from rigging.network import interface_for_ipv4
from rigging.timing import Duration, ExponentialBackoff

from marin.inference.config import ServingGeometry
from marin.inference.vllm_cache import VllmCompilationCache, VllmCompilationCacheReference

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30 * 60.0
_POLL_SECONDS = 1.0
_MAX_POLL_SECONDS = 30.0
_MASTER_PORT = 29500
# A follower restore downloads one archive under a 120 s transfer deadline, then extracts it.
_CACHE_SHARE_TIMEOUT_SECONDS = 180.0
_VLLM_HOST_IP_ENV = "VLLM_HOST_IP"
_GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME"


class _GangState(StrEnum):
    WAITING = "waiting"
    START_REQUESTED = "start_requested"
    STOPPING = "stopping"


class VllmCoordinatorActor:
    """Leader-owned, latched lifecycle state for an Iris vLLM gang."""

    def __init__(self, vllm_primary_address: str):
        self._vllm_primary_address = vllm_primary_address
        self._state = _GangState.WAITING
        self._stopped_followers: set[int] = set()
        self._compilation_cache_shared = False
        self._compilation_cache_reference: str | None = None
        self._compilation_cache_restores: dict[int, bool] = {}
        self._lock = threading.Lock()

    def vllm_primary_address(self) -> str:
        return self._vllm_primary_address

    def request_start(self) -> None:
        with self._lock:
            if self._state is _GangState.WAITING:
                self._state = _GangState.START_REQUESTED

    def state(self) -> _GangState:
        with self._lock:
            return self._state

    def request_shutdown(self) -> None:
        with self._lock:
            self._state = _GangState.STOPPING

    def follower_stopped(self, task_index: int) -> None:
        with self._lock:
            self._stopped_followers.add(task_index)

    def followers_stopped(self, task_indices: tuple[int, ...]) -> bool:
        with self._lock:
            return all(task_index in self._stopped_followers for task_index in task_indices)

    def share_compilation_cache(self, reference_json: str | None) -> None:
        """Latch the leader's cache reference JSON; ``None`` tells followers to skip restore."""
        with self._lock:
            self._compilation_cache_shared = True
            self._compilation_cache_reference = reference_json

    def compilation_cache(self) -> tuple[bool, str | None]:
        """Return whether the leader shared its cache, and the reference JSON if any."""
        with self._lock:
            return self._compilation_cache_shared, self._compilation_cache_reference

    def compilation_cache_restored(self, task_index: int, restored: bool) -> None:
        with self._lock:
            self._compilation_cache_restores[task_index] = restored

    def compilation_cache_restores(self) -> dict[int, bool]:
        with self._lock:
            return dict(self._compilation_cache_restores)


class IrisGangIdentity(Protocol):
    @property
    def coordinator_name(self) -> str: ...

    @property
    def task_index(self) -> int: ...

    @property
    def num_tasks(self) -> int: ...


@dataclass(frozen=True)
class IrisGangCoordinator:
    coordinator_name: str
    task_index: int
    num_tasks: int


@dataclass(frozen=True)
class IrisVllmLaunch:
    """Native-vLLM arguments, environment, and Iris task identity."""

    task_index: int
    num_tasks: int
    extra_cli_args: tuple[str, ...]
    host_ip: str
    gloo_interface: str
    coordinator_name: str
    tensor_parallel_size: int
    data_parallel_size: int

    @property
    def is_leader(self) -> bool:
        return self.task_index == 0

    @property
    def subprocess_env(self) -> dict[str, str]:
        return {
            _VLLM_HOST_IP_ENV: self.host_ip,
            _GLOO_SOCKET_IFNAME_ENV: self.gloo_interface,
        }


@contextlib.contextmanager
def iris_vllm_launch(geometry: ServingGeometry) -> Iterator[IrisVllmLaunch]:
    """Bootstrap native vLLM pipeline stages or a distributed data-parallel group."""
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("Iris vLLM launch must run inside an Iris job")
    if job_info.num_tasks != geometry.task_count:
        raise ValueError(f"vLLM geometry requires {geometry.task_count} Iris tasks, got {job_info.num_tasks}")

    endpoint_namespace = str(job_info.job_id).strip("/").replace("/", "-")
    coordinator_name = f"{endpoint_namespace}-vllm-coordinator"
    identity = IrisGangCoordinator(coordinator_name, job_info.task_index, job_info.num_tasks)
    with iris_gang_coordinator(identity, job_info.advertise_host, job_info.advertise_host) as master_addr:
        yield _build_task_launch(
            job_info,
            geometry=geometry,
            master_addr=master_addr,
            coordinator_name=coordinator_name,
        )


@contextlib.contextmanager
def iris_gang_coordinator(identity: IrisGangIdentity, advertise_host: str, primary_address: str) -> Iterator[str]:
    """Register the leader's address or discover it from a follower."""
    with contextlib.ExitStack() as stack:
        if identity.task_index == 0:
            server = ActorServer(host="0.0.0.0", port=0)
            server.register(identity.coordinator_name, VllmCoordinatorActor(primary_address))
            actor_port = server.serve_background()
            actor_address = f"http://{advertise_host}:{actor_port}"
            stack.callback(server.stop)
            stack.enter_context(iris_ctx().registry.registered(identity.coordinator_name, actor_address))
            logger.info("vLLM coordinator actor serving at %s", actor_address)
        else:
            client = _coordinator_client(identity.coordinator_name)
            primary_address = ""

            def coordinator_available() -> bool:
                nonlocal primary_address
                primary_address = client.vllm_primary_address()
                return True

            _wait_until(coordinator_available, error_message="Timed out waiting for the vLLM coordinator")
        yield primary_address


def start_iris_vllm_followers(launch: IrisGangIdentity) -> None:
    """Release followers that are waiting for an explicit startup decision."""
    if launch.task_index != 0:
        raise RuntimeError("Only the vLLM leader may request follower startup")
    client = _coordinator_client(launch.coordinator_name)

    def request_start() -> bool:
        client.request_start()
        return True

    _wait_until(request_start, error_message="Timed out requesting vLLM follower startup")


def wait_for_iris_vllm_start_decision(launch: IrisGangIdentity) -> bool:
    """Return whether to launch vLLM once the leader starts or skips the server."""
    if launch.task_index == 0:
        raise RuntimeError("The vLLM leader does not wait for follower startup")
    state = _wait_for_coordinator_state(launch, (_GangState.START_REQUESTED, _GangState.STOPPING))
    return state is _GangState.START_REQUESTED


def _wait_for_coordinator_state(
    launch: IrisGangIdentity,
    states: tuple[_GangState, ...],
    *,
    check_alive: Callable[[], None] | None = None,
    maximum_poll_seconds: float = _MAX_POLL_SECONDS,
) -> _GangState:
    client = _coordinator_client(launch.coordinator_name)
    state = _GangState.WAITING

    def refresh_state() -> bool:
        nonlocal state
        if check_alive is not None:
            check_alive()
        state = client.state()
        return True

    polling = ExponentialBackoff(initial=_POLL_SECONDS, maximum=maximum_poll_seconds)
    while True:
        _wait_until(refresh_state, error_message="Timed out contacting the vLLM leader")
        if state in states:
            return state
        time.sleep(polling.next_interval())


def wait_for_iris_vllm_shutdown(
    launch: IrisGangIdentity,
    check_alive: Callable[[], None],
    *,
    maximum_poll_seconds: float = _MAX_POLL_SECONDS,
) -> None:
    """Keep a follower alive until the leader requests final shutdown."""
    if launch.task_index == 0:
        raise RuntimeError("The vLLM leader does not wait for follower shutdown")
    _wait_for_coordinator_state(
        launch, (_GangState.STOPPING,), check_alive=check_alive, maximum_poll_seconds=maximum_poll_seconds
    )
    logger.info("vLLM follower %d received shutdown", launch.task_index)


def notify_iris_vllm_stopped(launch: IrisGangIdentity) -> None:
    """Tell the leader that this follower has stopped."""
    if launch.task_index == 0:
        raise RuntimeError("The vLLM leader does not send a follower stop acknowledgement")
    client = _coordinator_client(launch.coordinator_name)

    def acknowledge_stop() -> bool:
        client.follower_stopped(launch.task_index)
        return True

    _wait_until(
        acknowledge_stop,
        error_message="Timed out acknowledging that the vLLM follower stopped",
    )


@contextlib.contextmanager
def iris_vllm_followers(launch: IrisGangIdentity) -> Iterator[None]:
    """Stop followers after success; leave them running on failure for Iris retries."""
    if launch.task_index != 0:
        raise RuntimeError("Only the vLLM leader may coordinate followers")
    if launch.num_tasks == 1:
        yield
        return

    client = _coordinator_client(launch.coordinator_name)
    yield

    def request_shutdown() -> bool:
        client.request_shutdown()
        return True

    _wait_until(
        request_shutdown,
        error_message="Timed out requesting vLLM follower shutdown",
    )
    follower_indices = tuple(range(1, launch.num_tasks))
    _wait_until(
        lambda: client.followers_stopped(follower_indices),
        error_message="Timed out stopping vLLM followers",
    )


def share_iris_vllm_compilation_cache(
    launch: IrisGangIdentity,
    reference: VllmCompilationCacheReference | None,
    *,
    timeout_seconds: float = _CACHE_SHARE_TIMEOUT_SECONDS,
) -> None:
    """Share the leader's restored cache and wait, bounded, for follower restores.

    The cache only saves compile time, so a timeout or follower failure logs a
    warning and lets the leader launch vLLM anyway.
    """
    if launch.task_index != 0:
        raise RuntimeError("Only the vLLM leader may share its compilation cache")
    if launch.num_tasks == 1:
        return
    follower_indices = tuple(range(1, launch.num_tasks))
    restores: dict[int, bool] = {}
    started = time.monotonic()
    deadline = started + timeout_seconds
    try:
        client = _coordinator_client(launch.coordinator_name)
        reference_json = reference.model_dump_json() if reference is not None else None

        def share() -> bool:
            client.share_compilation_cache(reference_json)
            return True

        _wait_until(
            share,
            error_message="Timed out sharing the vLLM compilation cache",
            timeout_seconds=timeout_seconds,
        )
        if reference is None:
            return

        def followers_reported() -> bool:
            nonlocal restores
            restores = client.compilation_cache_restores()
            return all(task_index in restores for task_index in follower_indices)

        _wait_until(
            followers_reported,
            error_message="Timed out waiting for vLLM followers to restore the compilation cache",
            timeout_seconds=max(deadline - time.monotonic(), 0.0),
        )
    except Exception as error:
        missing = [task_index for task_index in follower_indices if task_index not in restores]
        logger.warning(
            "Launching vLLM before follower compilation cache restores finished; missing=%s reason=%s",
            missing,
            error,
        )
        return
    missed = [task_index for task_index in follower_indices if not restores[task_index]]
    if missed:
        logger.warning("vLLM followers %s missed the compilation cache key=%s", missed, reference.cache_key)
    logger.info(
        "vLLM followers restored the compilation cache key=%s restored=%d/%d wait=%.1fs",
        reference.cache_key,
        len(follower_indices) - len(missed),
        len(follower_indices),
        time.monotonic() - started,
    )


def restore_iris_vllm_compilation_cache(
    launch: IrisGangIdentity,
    check_alive: Callable[[], None],
    *,
    maximum_poll_seconds: float = _MAX_POLL_SECONDS,
) -> VllmCompilationCache | None:
    """Restore the leader's managed cache on this follower host before vLLM launches.

    Wait until the leader shares its cache or requests shutdown, restore it, and
    acknowledge. The caller closes the returned restore-only cache after vLLM stops.
    """
    if launch.task_index == 0:
        raise RuntimeError("The vLLM leader does not restore a follower compilation cache")
    client = _coordinator_client(launch.coordinator_name)
    shared: tuple[bool, str | None] = (False, None)
    state = _GangState.WAITING

    def refresh() -> bool:
        nonlocal shared, state
        check_alive()
        shared = client.compilation_cache()
        state = client.state()
        return True

    polling = ExponentialBackoff(initial=_POLL_SECONDS, maximum=maximum_poll_seconds)
    while True:
        _wait_until(refresh, error_message="Timed out contacting the vLLM leader")
        is_shared, reference_json = shared
        if is_shared:
            break
        if state is _GangState.STOPPING:
            return None
        time.sleep(polling.next_interval())
    if reference_json is None:
        logger.info("vLLM leader has no managed compilation cache; follower %d skips restore", launch.task_index)
        return None

    cache: VllmCompilationCache | None = None
    try:
        cache = VllmCompilationCache.restore_reference(VllmCompilationCacheReference.model_validate_json(reference_json))
    except Exception:
        logger.warning("vLLM follower %d failed to restore the compilation cache", launch.task_index, exc_info=True)
    restored = cache is not None and cache.restored

    def acknowledge_restore() -> bool:
        client.compilation_cache_restored(launch.task_index, restored)
        return True

    try:
        _wait_until(
            acknowledge_restore,
            error_message="Timed out acknowledging the vLLM compilation cache restore",
        )
    except BaseException:
        if cache is not None:
            cache.close()
        raise
    return cache


def _build_task_launch(
    job_info: JobInfo,
    *,
    geometry: ServingGeometry,
    master_addr: str,
    coordinator_name: str,
) -> IrisVllmLaunch:
    local_dp = geometry.local_data_parallel_size
    args = (
        "--tensor-parallel-size",
        str(geometry.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(geometry.pipeline_parallel_size),
        "--data-parallel-size",
        str(geometry.data_parallel_size),
        "--data-parallel-size-local",
        str(local_dp),
        "--nnodes",
        str(job_info.num_tasks),
        "--node-rank",
        str(job_info.task_index),
        "--master-addr",
        master_addr,
        "--master-port",
        str(_MASTER_PORT),
        "--device-ids",
        ",".join(str(index) for index in range(geometry.gpus_per_task)),
    )
    if local_dp != geometry.data_parallel_size:
        # Native --nnodes/--node-rank infer each host's DP rank. Leaving the
        # start-rank unset keeps rank zero's internal load balancer global.
        args = (
            *args,
            "--data-parallel-address",
            master_addr,
            "--data-parallel-rpc-port",
            str(_MASTER_PORT + 1),
            "--data-parallel-backend",
            "mp",
        )
    elif geometry.data_parallel_size > 1:
        args = (*args, "--data-parallel-start-rank", "0")
    if job_info.task_index != 0:
        args = (*args, "--headless")
    host_ip, gloo_interface = _node_network(job_info)
    return IrisVllmLaunch(
        task_index=job_info.task_index,
        num_tasks=job_info.num_tasks,
        extra_cli_args=args,
        host_ip=host_ip,
        gloo_interface=gloo_interface,
        coordinator_name=coordinator_name,
        tensor_parallel_size=geometry.tensor_parallel_size,
        data_parallel_size=geometry.data_parallel_size,
    )


def _node_network(job_info: JobInfo) -> tuple[str, str]:
    host_ip = os.environ.get(_VLLM_HOST_IP_ENV, job_info.advertise_host)
    gloo_interface = os.environ.get(_GLOO_SOCKET_IFNAME_ENV) or interface_for_ipv4(host_ip)
    logger.info("vLLM node network: host_ip=%s gloo_interface=%s", host_ip, gloo_interface)
    return host_ip, gloo_interface


def _coordinator_client(name: str) -> ActorClient:
    return ActorClient(iris_ctx().resolver, name, call_timeout=30.0, max_call_attempts=1)


def _wait_until(
    predicate: Callable[[], bool],
    *,
    error_message: str,
    timeout_seconds: float = _TIMEOUT_SECONDS,
) -> None:
    def retry_transient_actor_error() -> bool:
        try:
            return predicate()
        except ConnectError as exc:
            # Older controllers can report a not-yet-registered endpoint as a
            # Connect NOT_FOUND or UNIMPLEMENTED instead of an empty result.
            if is_retryable_error(exc) or exc.code in (Code.NOT_FOUND, Code.UNIMPLEMENTED):
                return False
            raise

    ExponentialBackoff(initial=_POLL_SECONDS, maximum=_MAX_POLL_SECONDS).wait_until_or_raise(
        retry_transient_actor_error,
        timeout=Duration.from_seconds(timeout_seconds),
        error_message=error_message,
    )
