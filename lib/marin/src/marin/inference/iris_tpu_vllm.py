# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate a Ray TPU cluster inside one Iris slice allocation."""

import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from fray.iris_backend import (
    IrisJobHandle,
    convert_constraints,
    convert_container_profile,
    convert_environment,
    convert_entrypoint,
    convert_resources,
    resolve_coscheduling,
)
from fray.types import JobRequest
from iris.actor.server import ActorServer
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info

from marin.inference import tpu_ray_runtime
from marin.inference.iris_vllm import VllmCoordinatorActor, _coordinator_client, _wait_until
from marin.inference.vllm_server import IsolatedTpuVllm

logger = logging.getLogger(__name__)
RAY_PORTS = ("ray-gcs", "ray-client", "ray-dashboard", "ray-dashboard-agent")


def submit_tpu_gang(request: JobRequest) -> IrisJobHandle:
    """Use Iris's named ports as well as Fray's TPU slice placement rules."""
    job = iris_ctx().client.submit(
        name=request.name,
        entrypoint=convert_entrypoint(request.entrypoint),
        resources=convert_resources(request.resources),
        environment=convert_environment(request.environment, request.resources.device),
        constraints=convert_constraints(request.resources),
        replicas=request.resources.replicas,
        coscheduling=resolve_coscheduling(request.resources, request.resources.replicas),
        ports=list(RAY_PORTS),
        max_retries_failure=request.max_retries_failure,
        max_retries_preemption=request.max_retries_preemption,
        max_task_failures=request.max_task_failures,
        task_image=request.resources.image,
        container_profile=convert_container_profile(request.resources),
        priority_band=request.priority,
        timeout=request.timeout,
    )
    return IrisJobHandle(job)


@dataclass(frozen=True)
class TpuRayGang:
    leader: bool
    address: str
    process: subprocess.Popen
    log_path: Path

    @property
    def subprocess_env(self) -> dict[str, str]:
        return {"RAY_ADDRESS": self.address, "TPU_MULTIHOST_BACKEND": "ray", "RAY_DEDUP_LOGS": "0"}

    def check_alive(self) -> None:
        if self.process.poll() is not None:
            detail = self.log_path.read_text()[-12000:] if self.log_path.exists() else ""
            raise RuntimeError(f"Ray supervisor exited {self.process.returncode}:\n{detail}")


def _stop_owned_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextlib.contextmanager
def tpu_ray_gang(launcher: IsolatedTpuVllm, *, chips_per_host: int, timeout_seconds: float) -> Iterator[TpuRayGang]:
    """Start Ray on every allocated host; only rank zero yields to the serving path.

    Followers wait for the leader to finish, then stop their own Ray node. Iris handles gang
    cancellation on failures; this context never stops a Ray process owned by another job.
    """
    info = get_job_info()
    if info is None or info.num_tasks < 2:
        raise ValueError("A multi-host TPU launch requires an Iris gang")
    ports = {name: info.ports[name] for name in RAY_PORTS}
    name = f"{str(info.job_id).strip('/').replace('/', '-')}-tpu-ray"
    ctx = iris_ctx()
    leader = info.task_index == 0
    os.environ["RAY_DEDUP_LOGS"] = "0"
    with contextlib.ExitStack() as stack:
        if leader:
            address = f"{info.advertise_host}:{ports['ray-gcs']}"
            actor = ActorServer(host="0.0.0.0", port=0)
            actor.register(name, VllmCoordinatorActor(address))
            actor_port = actor.serve_background()
            stack.callback(actor.stop)
            stack.enter_context(ctx.registry.registered(name, f"http://{info.advertise_host}:{actor_port}"))
        else:
            address = ""

            def find_head() -> bool:
                nonlocal address
                address = _coordinator_client(name).vllm_primary_address()
                return True

            _wait_until(find_head, error_message="Timed out discovering the TPU Ray head")
        directory = Path(stack.enter_context(TemporaryDirectory(prefix="marin-ray-")))
        ready_path = directory / "ready.json"
        ray_log = directory / "ray-start.log"
        config_path = directory / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "host": info.advertise_host,
                    "address": address,
                    "leader": leader,
                    "ports": ports,
                    "chips": chips_per_host,
                    "hosts": info.num_tasks,
                    "ready_path": str(ready_path),
                    "log_path": str(ray_log),
                    "timeout_seconds": timeout_seconds,
                }
            )
        )
        command = [
            *launcher.command()[:-1],
            "python",
            str(Path(tpu_ray_runtime.__file__).resolve()),
            "--config",
            str(config_path),
        ]
        environment = {
            **os.environ,
            **launcher.env(),
            "RAY_DEDUP_LOGS": "0",
            "RAY_USAGE_STATS_ENABLED": "0",
            "TPU_MULTIHOST_BACKEND": "ray",
            "RAY_ADDRESS": address,
        }
        supervisor_log = directory / "supervisor.log"
        with supervisor_log.open("w") as log:
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        stack.callback(_stop_owned_process, process)
        gang = TpuRayGang(leader, address, process, supervisor_log)
        deadline = time.monotonic() + timeout_seconds
        try:
            while not ready_path.exists():
                gang.check_alive()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out starting the TPU Ray gang")
                time.sleep(1)
            gang.check_alive()
            logger.info("TPU Ray gang ready: %d hosts, %d chips/host", info.num_tasks, chips_per_host)
            coordinator = _coordinator_client(name)
            if leader:
                yield gang
                coordinator.request_shutdown()
                _wait_until(
                    lambda: coordinator.followers_stopped(tuple(range(1, info.num_tasks))),
                    error_message="Timed out stopping TPU Ray followers",
                )
            else:
                while not coordinator.shutdown_requested():
                    gang.check_alive()
                    time.sleep(1)
                _stop_owned_process(process)
                coordinator.follower_stopped(info.task_index)
                yield gang
        except BaseException:
            for path in (supervisor_log, ray_log):
                if path.exists():
                    logger.error("%s:\n%s", path.name, path.read_text()[-12000:])
            raise
