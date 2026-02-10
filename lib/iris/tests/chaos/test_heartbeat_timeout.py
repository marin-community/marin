# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker heartbeat timeout detection and recovery.

Tests verify that workers correctly detect when the controller stops sending
heartbeats and reset their state appropriately.

This extends the chaos test suite with specific tests for heartbeat timeout.
"""

import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from .conftest import submit, wait


@pytest.mark.chaos
def test_worker_survives_transient_heartbeat_delay(cluster):
    """Worker doesn't reset on brief heartbeat delays.

    Verify that delays less than the timeout don't trigger a reset.
    Uses chaos injection to delay heartbeat by ~1s a few times, which
    stays well under the default 60s timeout. The job should complete
    successfully without the worker resetting.
    """
    _url, client = cluster

    def quick_job():
        return 42

    job = submit(client, quick_job, "transient-delay")

    # Inject transient delays (1s delay, only 3 times)
    # This simulates network hiccups but stays well under the 60s timeout
    enable_chaos("worker.heartbeat", delay_seconds=1.0, max_failures=3)

    # Wait for job to complete
    status = wait(client, job, timeout=30)

    # Verify job succeeded (worker didn't reset)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
