# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate an Iris task gang into vLLM's native multiprocess arguments."""

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.actor.client import ActorClient
from iris.actor.server import ActorServer
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.rpc.errors import is_retryable_error
from rigging.network import interface_for_ipv4
from rigging.timing import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30 * 60.0
_POLL_SECONDS = 1.0
_MASTER_PORT = 29500
_VLLM_HOST_IP_ENV = "VLLM_HOST_IP"
_GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME"


class VllmCoordinatorActor:
    """Leader-owned, latched lifecycle state for a native vLLM gang."""

    def __init__(self, vllm_primary_address: str):
        self._vllm_primary_address = vllm_primary_address
        self._shutdown_requested = False
        self._stopped_followers: set[int] = set()
        self._lock = threading.Lock()

    def vllm_primary_address(self) -> str:
        return self._vllm_primary_address

    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_requested

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutdown_requested = True

    def follower_stopped(self, task_index: int) -> None:
        with self._lock:
            self._stopped_followers.add(task_index)

    def followers_stopped(self, task_indices: tuple[int, ...]) -> bool:
        with self._lock:
            return all(task_index in self._stopped_followers for task_index in task_indices)


@dataclass(frozen=True)
class IrisVllmLaunch:
    """Native-vLLM arguments, environment, and Iris task identity."""

    task_index: int
    num_tasks: int
    extra_cli_args: tuple[str, ...]
    host_ip: str
    gloo_interface: str
    coordinator_name: str

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
def iris_vllm_launch(
    *,
    pipeline_parallel_size: int,
    data_parallel_size: int,
) -> Iterator[IrisVllmLaunch]:
    """Bootstrap a stage-striped native-vLLM task from Iris metadata."""
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("Iris vLLM launch must run inside an Iris job")
    if job_info.num_tasks != pipeline_parallel_size:
        raise ValueError(
            f"vLLM PP={pipeline_parallel_size} requires {pipeline_parallel_size} Iris tasks, got {job_info.num_tasks}"
        )

    endpoint_namespace = str(job_info.job_id).strip("/").replace("/", "-")
    coordinator_name = f"{endpoint_namespace}-vllm-coordinator"
    ctx = iris_ctx()

    with contextlib.ExitStack() as stack:
        if job_info.task_index == 0:
            master_addr = job_info.advertise_host
            server = ActorServer(host="0.0.0.0", port=0)
            server.register(coordinator_name, VllmCoordinatorActor(master_addr))
            actor_port = server.serve_background()
            actor_address = f"http://{job_info.advertise_host}:{actor_port}"
            stack.callback(server.stop)
            stack.enter_context(ctx.registry.registered(coordinator_name, actor_address))
            logger.info("vLLM coordinator actor serving at %s", actor_address)
        else:
            client = _coordinator_client(coordinator_name)
            master_addr: str | None = None

            def coordinator_available() -> bool:
                nonlocal master_addr
                master_addr = client.vllm_primary_address()
                return True

            _wait_until(
                coordinator_available,
                error_message="Timed out waiting for the vLLM coordinator",
            )
            assert master_addr is not None

        yield _build_task_launch(
            job_info,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            master_addr=master_addr,
            coordinator_name=coordinator_name,
        )


def wait_for_iris_vllm_shutdown(
    launch: IrisVllmLaunch,
    check_alive: Callable[[], None],
) -> None:
    """Keep a follower alive until the leader requests final shutdown."""
    if launch.is_leader:
        raise RuntimeError("The vLLM leader does not wait for follower shutdown")
    client = _coordinator_client(launch.coordinator_name)

    def shutdown_requested() -> bool:
        if client.shutdown_requested():
            return True
        check_alive()
        return False

    _wait_until(
        shutdown_requested,
        error_message="Timed out waiting for the vLLM leader to finish",
    )
    logger.info("vLLM follower %d received shutdown", launch.task_index)


def notify_iris_vllm_stopped(launch: IrisVllmLaunch) -> None:
    """Tell the leader that this follower has stopped."""
    if launch.is_leader:
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
def iris_vllm_followers(launch: IrisVllmLaunch) -> Iterator[None]:
    """Request follower shutdown when the leader's workload is finished."""
    if not launch.is_leader:
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


def _build_task_launch(
    job_info: JobInfo,
    *,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    master_addr: str,
    coordinator_name: str,
) -> IrisVllmLaunch:
    args = (
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
        "--data-parallel-size",
        str(data_parallel_size),
        "--data-parallel-size-local",
        str(data_parallel_size),
        "--data-parallel-start-rank",
        "0",
        "--nnodes",
        str(pipeline_parallel_size),
        "--node-rank",
        str(job_info.task_index),
        "--master-addr",
        master_addr,
        "--master-port",
        str(_MASTER_PORT),
        "--device-ids",
        ",".join(str(index) for index in range(data_parallel_size)),
    )
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

    ExponentialBackoff(initial=_POLL_SECONDS, maximum=30.0).wait_until_or_raise(
        retry_transient_actor_error,
        timeout=Duration.from_seconds(_TIMEOUT_SECONDS),
        error_message=error_message,
    )
