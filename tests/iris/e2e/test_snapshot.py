# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot checkpoint E2E tests.

Tests the BeginCheckpoint RPC and edge cases like worker failure after checkpoint.
Restore is startup-only and tested via unit tests in test_snapshot_cli.py.
"""

import time

import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2

from .helpers import _quick

pytestmark = pytest.mark.e2e


def test_checkpoint_returns_snapshot_metadata(cluster):
    """BeginCheckpoint RPC returns a valid snapshot path and counts."""
    job = cluster.submit(_quick, "pre-checkpoint")
    cluster.wait(job, timeout=30)

    resp = cluster.controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    assert resp.snapshot_path
    assert resp.created_at.epoch_ms > 0
    assert resp.job_count >= 1


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
