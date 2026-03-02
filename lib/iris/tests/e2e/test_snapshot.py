# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot checkpoint/restore E2E tests.

Tests the full checkpoint/restore cycle through the controller RPCs,
including edge cases like worker failure after restore and endpoint survival.
"""

import time
import uuid

import pytest
from iris.chaos import enable_chaos
from iris.cluster.client import get_job_info
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

from .helpers import _quick, _slow

pytestmark = pytest.mark.e2e


def test_checkpoint_returns_snapshot_metadata(cluster):
    """BeginCheckpoint RPC returns a valid snapshot path and counts."""
    job = cluster.submit(_quick, "pre-checkpoint")
    cluster.wait(job, timeout=30)

    resp = cluster.controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    assert resp.snapshot_path
    assert resp.created_at.epoch_ms > 0
    assert resp.job_count >= 1


def test_checkpoint_preserves_running_job(cluster):
    """A running job is present in the checkpoint and survives load_checkpoint."""
    # Submit a long-running job so it's RUNNING during checkpoint
    job = cluster.submit(_slow, "long-runner")
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    # Checkpoint
    ckpt_resp = cluster.controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    assert ckpt_resp.job_count >= 1
    assert ckpt_resp.worker_count >= 1

    # Load checkpoint into fresh state (this is additive, but verifies the RPC works)
    load_resp = cluster.controller_client.load_checkpoint(cluster_pb2.Controller.LoadCheckpointRequest())
    assert load_resp.jobs_restored >= 1
    assert load_resp.workers_restored >= 1

    # The original job should still be tracked
    status = cluster.status(job)
    assert status.state in (
        cluster_pb2.JOB_STATE_RUNNING,
        cluster_pb2.JOB_STATE_SUCCEEDED,
    )


def test_checkpoint_restore_with_worker_death(cluster):
    """Worker dies after checkpoint; task is retried when heartbeats fail.

    Simulates the "worker dies during restart window" scenario:
    1. Submit a job, wait for it to start running
    2. Checkpoint
    3. Inject chaos to kill heartbeats (simulating worker death)
    4. The controller detects worker failure and retries the task
    """

    def medium_job():
        time.sleep(5)
        return 42

    job = cluster.submit(medium_job, "worker-death-retry", max_retries_preemption=10)
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    # Checkpoint while job is running
    ckpt_resp = cluster.controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    assert ckpt_resp.job_count >= 1

    # Simulate worker death: heartbeats fail for enough rounds to trigger
    # worker failure. Local config uses threshold=3.
    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=4,
        delay_seconds=0.01,
    )

    # The job should eventually succeed after the worker recovers and the
    # task is retried on a fresh worker.
    status = cluster.wait(job, timeout=45)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def _register_and_sleep(prefix):
    """Register an endpoint then sleep so the job stays running for checkpoint."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
    try:
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=f"{prefix}/actor",
            address="localhost:9999",
            job_id=info.job_id.to_wire(),
            metadata={"role": "primary"},
        )
        client.register_endpoint(request)
        time.sleep(30)
    finally:
        client.close()


def test_checkpoint_restore_preserves_endpoints(cluster):
    """Endpoints registered by actor jobs survive checkpoint/load_checkpoint."""
    prefix = f"snap-ep-{uuid.uuid4().hex[:8]}"
    job = cluster.submit(_register_and_sleep, "endpoint-snap", prefix)
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    # Wait for the endpoint to be registered
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        list_resp = cluster.controller_client.list_endpoints(
            cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{prefix}/")
        )
        if len(list_resp.endpoints) > 0:
            break
        time.sleep(0.5)
    assert len(list_resp.endpoints) == 1, "Endpoint not registered before checkpoint"

    # Checkpoint
    cluster.controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())

    # Load checkpoint (additive — endpoints should still be present)
    load_resp = cluster.controller_client.load_checkpoint(cluster_pb2.Controller.LoadCheckpointRequest())
    assert load_resp.jobs_restored >= 1

    # Verify endpoints survived
    list_resp = cluster.controller_client.list_endpoints(
        cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{prefix}/")
    )
    assert len(list_resp.endpoints) >= 1
    assert list_resp.endpoints[0].metadata["role"] == "primary"
