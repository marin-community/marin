# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from pathlib import Path
from iris.chaos import reset_chaos
from iris.cluster.vm.cluster_manager import ClusterManager
from iris.cluster.vm.config import make_local_config
from iris.cluster.vm.config import load_config
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


@pytest.fixture
def cluster():
    """Boots a local cluster via ClusterManager, yields (url, client)."""
    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config)
    manager = ClusterManager(config)
    with manager.connect() as url:
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
