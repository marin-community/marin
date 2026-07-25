# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for retry behavior across task attempts.

These tests need real job execution to verify chaos injection behavior.
"""

import uuid

import pytest
from iris.chaos import enable_chaos
from iris.rpc import job_pb2

pytestmark = [pytest.mark.requires_cluster, pytest.mark.timeout(60)]


def test_superseding_attempt_logs_info(cluster):
    """Verify job succeeds after chaos injection fails container creation on first attempt.

    Uses chaos injection to fail container creation once, forcing retry.
    The second attempt should succeed normally.
    """
    run_id = uuid.uuid4().hex[:8]

    enable_chaos(
        "worker.create_container",
        failure_rate=1.0,
        max_failures=1,
        error=RuntimeError("chaos: container creation failed"),
    )

    job = cluster.submit(
        lambda: "ok",
        f"supersede-{run_id}",
        max_retries_failure=2,
        max_task_failures=2,
    )

    status = cluster.wait(job, timeout=60)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED, f"Job should succeed after retry: {status}"
