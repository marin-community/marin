# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from pathlib import Path
from iris.chaos import reset_chaos
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.rpc import config_pb2
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec, EnvironmentSpec, is_job_finished

from .chronos import VirtualClock

IRIS_ROOT = Path(__file__).resolve().parents[2]  # lib/iris (from tests/chaos/conftest.py -> tests -> lib/iris)
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "demo.yaml"


def _quick():
    return 1


def _slow():
    import time

    time.sleep(120)


def _block(s):
    """Block until sentinel is signalled. Pass a SentinelFile instance."""
    s.wait()


@pytest.fixture(autouse=True)
def _reset_chaos():
    yield
    reset_chaos()


@pytest.fixture
def chronos(monkeypatch):
    """Virtual time fixture - makes time.sleep() controllable for fast tests."""
    clock = VirtualClock()

    # Patch time module
    monkeypatch.setattr(time, "time", clock.time)
    monkeypatch.setattr(time, "monotonic", clock.time)
    monkeypatch.setattr(time, "sleep", clock.sleep)

    return clock


def _add_coscheduling_group(config: config_pb2.IrisClusterConfig) -> None:
    """Add a scale group with num_vms=2 so coscheduling tests can find a match.

    v5litepod-16 has vm_count=2, so the local platform creates 2 workers per slice
    sharing the same tpu-name. Setting num_vms=2 lets the demand router match
    coscheduled jobs with replicas=2.
    """
    sg = config.scale_groups["tpu_cosched_2"]
    sg.name = "tpu_cosched_2"
    sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.accelerator_variant = "v5litepod-16"
    sg.num_vms = 2
    sg.min_slices = 1
    sg.max_slices = 2
    sg.resources.cpu = 128
    sg.resources.memory_bytes = 128 * 1024 * 1024 * 1024
    sg.resources.disk_bytes = 1024 * 1024 * 1024 * 1024
    sg.slice_template.preemptible = True
    sg.slice_template.num_vms = 2
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.slice_template.accelerator_variant = "v5litepod-16"


@pytest.fixture
def cluster():
    """Boots a local cluster, yields (url, client)."""
    config = load_config(DEFAULT_CONFIG)
    _add_coscheduling_group(config)
    config = make_local_config(config)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        yield url, client


def submit(client, fn, name, *args, **kw):
    return client.submit(
        entrypoint=Entrypoint.from_callable(fn, *args),
        name=name,
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
        **kw,
    )


def wait(client, job, timeout=60, chronos=None):
    """Wait for job to reach terminal state.

    If chronos is provided, uses virtual time.
    Otherwise uses real time.sleep().
    """
    if chronos is not None:
        # Virtual time: tick until job completes or timeout
        start_time = chronos.time()
        while chronos.time() - start_time < timeout:
            status = client.status(job.job_id)
            if is_job_finished(status.state):
                return status
            chronos.tick(0.5)  # Advance by poll interval
        return client.status(job.job_id)
    else:
        # Real time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = client.status(job.job_id)
            if is_job_finished(status.state):
                return status
            time.sleep(0.5)
        return client.status(job.job_id)
