# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""High task count tests to expose race conditions in concurrent scheduling.

Migrated from tests/cluster/test_e2e.py::TestHighTaskCount.
"""

import time

import pytest
from iris.chaos import enable_chaos, reset_chaos
from iris.rpc import cluster_pb2

pytestmark = pytest.mark.e2e


@pytest.mark.slow
def test_128_tasks_concurrent_scheduling(multi_worker_cluster, sentinel):
    """128 simultaneous tasks expose heartbeat iteration race conditions.

    Reproduces the bug where _heartbeat_worker() iterates over
    worker.running_tasks without holding state._lock while the scheduling
    thread modifies the set, causing "Set changed size during iteration".
    """
    enable_chaos("controller.heartbeat.iteration", delay_seconds=0.01)

    def waiting_task(s):
        from iris.time_utils import Duration

        s.wait(timeout=Duration.from_seconds(2))
        return "done"

    try:
        job = multi_worker_cluster.submit(
            waiting_task,
            "race-test",
            sentinel,
            cpu=0,
            replicas=128,
        )

        time.sleep(1.0)
        sentinel.signal()

        status = multi_worker_cluster.wait(job, timeout=120)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job failed: {status}"
    finally:
        reset_chaos()
