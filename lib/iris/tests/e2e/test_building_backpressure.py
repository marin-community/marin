# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backpressure test: slow-starting containers must not exceed max_building_tasks_per_worker."""

from pathlib import Path

import pytest
from iris.client.client import IrisClient
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.rpc import config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

from .conftest import IrisTestCluster

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
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(1, timeout=30)
        yield tc
        controller_client.close()
