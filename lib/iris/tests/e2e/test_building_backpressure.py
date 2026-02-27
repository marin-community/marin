# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backpressure test: slow-starting containers must not exceed max_building_tasks_per_worker."""

import time
from pathlib import Path

import pytest
from iris.client.client import IrisClient
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.cluster.runtime.process import ProcessContainerHandle
from iris.cluster.runtime.types import ContainerPhase, ContainerStatus
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

from .conftest import TestCluster
from .helpers import _quick

pytestmark = pytest.mark.e2e

IRIS_ROOT = Path(__file__).resolve().parents[2]
READY_DELAY = 3.0
NUM_TASKS = 20
MAX_BUILDING = 4  # default max_building_tasks_per_worker


@pytest.fixture
def single_worker_cluster():
    """Single-worker cluster so building backpressure is observable."""
    config = load_config(IRIS_ROOT / "examples" / "demo.yaml")
    config.scale_groups.clear()
    sg = config.scale_groups["local-cpu"]
    sg.name = "local-cpu"
    sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.num_vms = 1
    sg.min_slices = 1
    sg.max_slices = 1
    sg.resources.cpu_millicores = 8000
    sg.resources.memory_bytes = 16 * 1024**3
    sg.resources.disk_bytes = 50 * 1024**3
    sg.slice_template.local.SetInParent()
    config = make_local_config(config)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = TestCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(1, timeout=30)
        yield tc
        controller_client.close()


def test_building_backpressure_with_slow_starting_containers(single_worker_cluster, monkeypatch):
    """With slow-starting containers, at most max_building_tasks_per_worker tasks
    should be in BUILDING/ASSIGNED at any time.

    Monkeypatches ProcessContainerHandle.status() to return ready=False for
    READY_DELAY seconds after run(), simulating K8S pod pending phase.
    """
    cluster = single_worker_cluster
    original_run = ProcessContainerHandle.run
    original_status = ProcessContainerHandle.status
    run_times: dict[int, float] = {}

    def patched_run(self):
        run_times[id(self)] = time.monotonic()
        return original_run(self)

    def patched_status(self) -> ContainerStatus:
        result = original_status(self)
        run_time = run_times.get(id(self))
        if run_time is not None and result.phase == ContainerPhase.RUNNING:
            elapsed = time.monotonic() - run_time
            if elapsed < READY_DELAY:
                result = ContainerStatus(
                    phase=ContainerPhase.PENDING,
                    exit_code=result.exit_code,
                    error=result.error,
                    error_kind=result.error_kind,
                    oom_killed=result.oom_killed,
                )
        return result

    monkeypatch.setattr(ProcessContainerHandle, "run", patched_run)
    monkeypatch.setattr(ProcessContainerHandle, "status", patched_status)

    jobs = [cluster.submit(_quick, f"burst-{i}", cpu=0) for i in range(NUM_TASKS)]

    peak_building = 0
    all_done = False
    deadline = time.monotonic() + 120

    while not all_done and time.monotonic() < deadline:
        building_count = 0
        all_done = True
        for job in jobs:
            status = cluster.status(job)
            for task in status.tasks:
                if task.state in (cluster_pb2.TASK_STATE_BUILDING, cluster_pb2.TASK_STATE_ASSIGNED):
                    building_count += 1
            if not status.state or status.state in (
                cluster_pb2.JOB_STATE_PENDING,
                cluster_pb2.JOB_STATE_RUNNING,
            ):
                all_done = False

        peak_building = max(peak_building, building_count)
        time.sleep(0.3)

    # Allow +1 slack for race between scheduling and heartbeat reporting
    assert peak_building <= MAX_BUILDING + 1, (
        f"Peak building count {peak_building} exceeded limit {MAX_BUILDING}. " f"Backpressure is not working."
    )

    # All jobs should eventually succeed
    for job in jobs:
        status = cluster.wait(job, timeout=120)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
