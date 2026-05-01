# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint/restore test for Iris controller."""

import time
from pathlib import Path

import pytest
from iris.cli.worker_health import query_workers
from iris.client.client import IrisClient, Job
from iris.cluster.config import load_config, make_local_config
from iris.cluster.providers.local.cluster import LocalCluster
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, is_job_finished
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

IRIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"


def _quick():
    return 1


class _IrisTestHelper:
    """Minimal helper to submit and wait for jobs (standalone, no integration fixtures)."""

    def __init__(self, url: str, client: IrisClient, controller_client: ControllerServiceClientSync):
        self.url = url
        self.client = client
        self.controller_client = controller_client

    def wait_for_workers(self, count: int, timeout: float = 30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            healthy = [w for w in query_workers(self.url) if w.healthy]
            if len(healthy) >= count:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Expected {count} healthy workers, timed out")

    def submit(self, fn, name: str) -> Job:
        return self.client.submit(
            entrypoint=Entrypoint.from_callable(fn),
            name=name,
            resources=ResourceSpec(cpu=1, memory="1g"),
            environment=EnvironmentSpec(),
        )

    def wait(self, job: Job, timeout: float = 30) -> job_pb2.JobStatus:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self.controller_client.get_job_status(
                controller_pb2.Controller.GetJobStatusRequest(job_id=job.job_id.to_wire())
            )
            if is_job_finished(resp.job.state):
                return resp.job
            time.sleep(0.5)
        raise TimeoutError(f"Job {job.job_id} did not finish within {timeout}s")


@pytest.mark.slow
def test_checkpoint_restore():
    """Controller restart resumes from checkpoint: completed jobs visible, cluster functional."""
    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config)

    cluster = LocalCluster(config)
    url = cluster.start()
    try:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = _IrisTestHelper(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(1, timeout=30)

        job = tc.submit(_quick, "pre-restart")
        tc.wait(job, timeout=30)
        saved_job_id = job.job_id.to_wire()

        ckpt = controller_client.begin_checkpoint(controller_pb2.Controller.BeginCheckpointRequest())
        assert ckpt.checkpoint_path, "begin_checkpoint returned empty path"
        assert ckpt.job_count >= 1
        controller_client.close()

        url = cluster.restart()

        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = _IrisTestHelper(
            url=url, client=IrisClient.remote(url, workspace=IRIS_ROOT), controller_client=controller_client
        )

        resp = controller_client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=saved_job_id))
        assert resp.job.state == job_pb2.JOB_STATE_SUCCEEDED, f"Pre-restart job has state {resp.job.state} after restore"

        tc.wait_for_workers(1, timeout=30)
        post_job = tc.submit(_quick, "post-restart")
        status = tc.wait(post_job, timeout=30)
        assert status.state == job_pb2.JOB_STATE_SUCCEEDED

        controller_client.close()
    finally:
        cluster.close()
