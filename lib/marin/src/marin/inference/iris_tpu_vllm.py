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

from iris.cluster.client.job_info import get_job_info

from marin.inference import tpu_ray_runtime
from marin.inference.iris_vllm import (
    IrisGangCoordinator,
    iris_gang_coordinator,
    iris_vllm_followers,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_server import IsolatedTpuVllm

logger = logging.getLogger(__name__)
RAY_PORTS = ("ray-gcs", "ray-client", "ray-dashboard", "ray-dashboard-agent")


@dataclass(frozen=True)
class TpuRayGang:
    leader: bool
    address: str
    process: subprocess.Popen
    log_path: Path

    @property
    def subprocess_env(self) -> dict[str, str]:
        return tpu_ray_runtime.ray_environment(self.address)

    def check_alive(self) -> None:
        if self.process.poll() is not None:
            detail = self.log_path.read_text()[-tpu_ray_runtime.RAY_LOG_TAIL_CHARS :] if self.log_path.exists() else ""
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
    """Start Ray on every allocated host and coordinate their shutdown.

    Followers wait for the leader to finish, then stop their own Ray node. Iris handles gang
    cancellation on failures; this context never stops a Ray process owned by another job.
    """
    info = get_job_info()
    if info is None or info.num_tasks < 2:
        raise ValueError("A multi-host TPU launch requires an Iris gang")
    ports = {name: info.ports[name] for name in RAY_PORTS}
    name = f"{str(info.job_id).strip('/').replace('/', '-')}-tpu-ray"
    identity = IrisGangCoordinator(name, info.task_index, info.num_tasks)
    leader = info.task_index == 0
    os.environ["RAY_DEDUP_LOGS"] = "0"
    with contextlib.ExitStack() as stack:
        address = stack.enter_context(
            iris_gang_coordinator(identity, info.advertise_host, f"{info.advertise_host}:{ports['ray-gcs']}")
        )
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
            **tpu_ray_runtime.ray_environment(address),
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
            if leader:
                with iris_vllm_followers(identity):
                    yield gang
            else:
                wait_for_iris_vllm_shutdown(identity, gang.check_alive, maximum_poll_seconds=1.0)
                _stop_owned_process(process)
                notify_iris_vllm_stopped(identity)
                yield gang
        except BaseException:
            for path in (supervisor_log, ray_log):
                if path.exists():
                    logger.error("%s:\n%s", path.name, path.read_text()[-tpu_ray_runtime.RAY_LOG_TAIL_CHARS :])
            raise
