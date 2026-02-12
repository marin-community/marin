# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Basic smoke tests for job lifecycle.

Migrated from tests/chaos/test_smoke.py and tests/cluster/test_e2e.py::TestJobLifecycle.
"""

import pytest
from iris.rpc import cluster_pb2

from .helpers import _block, _failing, _quick

pytestmark = pytest.mark.e2e


def test_submit_and_succeed(cluster):
    job = cluster.submit(_quick, "smoke-succeed")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_failure_propagates(cluster):
    job = cluster.submit(_failing, "smoke-fail")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


def test_kill_running_job(cluster, sentinel):
    job = cluster.submit(_block, "smoke-kill", sentinel)
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=10)
    cluster.kill(job)
    sentinel.signal()
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_KILLED


def test_job_with_args(cluster):
    def add(a, b):
        return a + b

    job = cluster.submit(add, "smoke-add", 10, 32)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_multiple_jobs_complete(cluster):
    def fast_job(n):
        return n * 2

    jobs = [cluster.submit(fast_job, f"smoke-multi-{i}", i) for i in range(5)]
    for job in jobs:
        status = cluster.wait(job, timeout=30)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
