# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate an Iris task gang into vLLM's native multiprocess arguments."""

import contextlib
import fcntl
import logging
import os
import socket
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.runtime.jax_init import poll_for_registered_endpoint
from rigging.timing import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

_COORDINATOR_ENDPOINT = "vllm-coordinator"
_READY_ENDPOINT_PREFIX = "vllm-ready-"
_SHUTDOWN_ENDPOINT = "vllm-shutdown"
_TIMEOUT_SECONDS = 30 * 60.0
_POLL_SECONDS = 1.0
_MASTER_PORT = 29500
_SIOCGIFADDR = 0x8915
_MISSING_ENDPOINT_CODES = frozenset({Code.NOT_FOUND, Code.UNIMPLEMENTED})
_VLLM_HOST_IP_ENV = "VLLM_HOST_IP"
_GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME"


@dataclass(frozen=True)
class IrisVllmLaunch:
    """Native-vLLM arguments, environment, and Iris task identity."""

    task_index: int
    num_tasks: int
    extra_cli_args: tuple[str, ...]
    host_ip: str
    gloo_interface: str
    endpoint_namespace: str

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
    """Bootstrap a stage-striped PP=1/2 native-vLLM task from Iris metadata."""
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("Iris vLLM launch must run inside an Iris job")
    if job_info.num_tasks != pipeline_parallel_size:
        raise ValueError(
            f"vLLM PP={pipeline_parallel_size} requires {pipeline_parallel_size} Iris tasks, got {job_info.num_tasks}"
        )

    endpoint_namespace = str(job_info.job_id).strip("/").replace("/", "-")
    coordinator_name = f"{endpoint_namespace}-{_COORDINATOR_ENDPOINT}"
    if job_info.task_index == 0:
        master_addr = job_info.advertise_host
        coordinator_registration = iris_ctx().registry.registered(coordinator_name, f"{master_addr}:{_MASTER_PORT}")
    else:
        address = poll_for_registered_endpoint(
            iris_ctx().resolver,
            coordinator_name,
            _TIMEOUT_SECONDS,
            _POLL_SECONDS,
        )
        master_addr, port = address.rsplit(":", maxsplit=1)
        if int(port) != _MASTER_PORT:
            raise ValueError(f"Unexpected vLLM coordinator address: {address!r}")
        coordinator_registration = contextlib.nullcontext()

    with coordinator_registration:
        yield _task_launch(
            job_info,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            master_addr=master_addr,
            endpoint_namespace=endpoint_namespace,
        )


def wait_for_iris_vllm_shutdown(
    launch: IrisVllmLaunch,
    check_alive: Callable[[], None],
) -> None:
    """Keep a verified follower alive until the leader finishes its workload."""
    if launch.is_leader:
        raise RuntimeError("The vLLM leader does not wait for its own shutdown signal")
    ready_name = _ready_endpoint_name(launch, launch.task_index)
    shutdown_name = _shutdown_endpoint_name(launch)

    def shutdown_requested() -> bool:
        check_alive()
        return _resolve_endpoint(shutdown_name) is not None

    with iris_ctx().registry.registered(ready_name, launch.host_ip):
        _wait_until(
            shutdown_requested,
            error_message="Timed out waiting for the vLLM leader to finish",
        )


@contextlib.contextmanager
def iris_vllm_followers(launch: IrisVllmLaunch) -> Iterator[None]:
    """Wait for verified followers, then stop them when the leader is done."""
    if not launch.is_leader:
        raise RuntimeError("Only the vLLM leader may coordinate followers")
    if launch.num_tasks == 1:
        yield
        return

    def all_followers_ready() -> bool:
        missing = [
            task_index
            for task_index in range(1, launch.num_tasks)
            if _resolve_endpoint(_ready_endpoint_name(launch, task_index)) is None
        ]
        if missing:
            logger.info("Waiting for vLLM followers %s", missing)
        return not missing

    _wait_until(
        all_followers_ready,
        error_message="Timed out waiting for vLLM followers",
    )
    try:
        yield
    finally:
        shutdown_name = _shutdown_endpoint_name(launch)
        with iris_ctx().registry.registered(shutdown_name, launch.host_ip):

            def all_followers_stopped() -> bool:
                return all(
                    _resolve_endpoint(_ready_endpoint_name(launch, task_index)) is None
                    for task_index in range(1, launch.num_tasks)
                )

            _wait_until(
                all_followers_stopped,
                error_message="Timed out stopping vLLM followers",
            )


def _task_launch(
    job_info: JobInfo,
    *,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    master_addr: str,
    endpoint_namespace: str,
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
        endpoint_namespace=endpoint_namespace,
    )


def _node_network(job_info: JobInfo) -> tuple[str, str]:
    host_ip = os.environ.get(_VLLM_HOST_IP_ENV, job_info.advertise_host)
    gloo_interface = os.environ.get(_GLOO_SOCKET_IFNAME_ENV) or _interface_for_ipv4(host_ip)
    logger.info("vLLM node network: host_ip=%s gloo_interface=%s", host_ip, gloo_interface)
    return host_ip, gloo_interface


def _interface_for_ipv4(address: str) -> str:
    """Find the local interface that owns ``address``."""
    packed_address = socket.inet_aton(socket.gethostbyname(address))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, interface in socket.if_nameindex():
            request = struct.pack("256s", interface.encode()[:15])
            try:
                interface_address = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, request)[20:24]
            except OSError:
                continue
            if interface_address == packed_address:
                return interface
    raise RuntimeError(f"No local network interface owns Iris advertise address {address!r}")


def _resolve_endpoint(name: str) -> str | None:
    try:
        result = iris_ctx().resolver.resolve(name)
    except ConnectError as exc:
        if exc.code in _MISSING_ENDPOINT_CODES:
            return None
        raise
    if result.is_empty:
        return None
    return result.first().url


def _wait_until(
    predicate: Callable[[], bool],
    *,
    error_message: str,
) -> None:
    def retry_transient_unavailability() -> bool:
        try:
            return predicate()
        except ConnectError as exc:
            if exc.code is Code.UNAVAILABLE:
                return False
            raise

    ExponentialBackoff(initial=_POLL_SECONDS, maximum=30.0).wait_until_or_raise(
        retry_transient_unavailability,
        timeout=Duration.from_seconds(_TIMEOUT_SECONDS),
        error_message=error_message,
    )


def _ready_endpoint_name(launch: IrisVllmLaunch, task_index: int) -> str:
    return f"{launch.endpoint_namespace}-{_READY_ENDPOINT_PREFIX}{task_index}"


def _shutdown_endpoint_name(launch: IrisVllmLaunch) -> str:
    return f"{launch.endpoint_namespace}-{_SHUTDOWN_ENDPOINT}"
