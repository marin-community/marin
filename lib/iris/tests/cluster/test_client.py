# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for job lifecycle operations through the RPC service layer.

These tests exercise the ControllerServiceImpl API parameterized across
both GCP and K8s providers via the ServiceTestHarness.
"""

import pytest
from connectrpc.errors import ConnectError

from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.rpc import controller_pb2

from .conftest import ServiceTestHarness


def _ensure_workers(harness: ServiceTestHarness) -> None:
    """Register a GCP worker if needed. K8s harness already has a default node pool."""
    if harness.provider_type == "gcp":
        harness.register_gcp_worker("w1")


def test_submit_rejects_duplicate_name(harness: ServiceTestHarness):
    """Second launch with the same name raises ALREADY_EXISTS."""
    harness.submit("dup-job")
    with pytest.raises(ConnectError, match="already exists"):
        harness.submit("dup-job")


def test_list_jobs_returns_all_jobs(harness: ServiceTestHarness):
    """All submitted jobs appear in list_jobs results."""
    id1 = harness.submit("list-job-1")
    id2 = harness.submit("list-job-2")

    resp = harness.service.list_jobs(controller_pb2.Controller.ListJobsRequest(), None)
    job_ids = {j.job_id for j in resp.jobs}

    assert id1.to_wire() in job_ids
    assert id2.to_wire() in job_ids


def test_list_jobs_filter_by_state(harness: ServiceTestHarness):
    """list_jobs state_filter narrows results to the requested state."""
    _ensure_workers(harness)

    # Drive one job to completion first, then submit a second job that
    # stays pending. This ordering avoids the K8s sync (triggered by
    # drive_job_to_completion) from also advancing the pending job.
    done_id = harness.submit("will-succeed")
    harness.drive_job_to_completion(done_id)
    pending_id = harness.submit("stays-pending")

    succeeded = harness.service.list_jobs(controller_pb2.Controller.ListJobsRequest(state_filter="succeeded"), None)
    succeeded_ids = {j.job_id for j in succeeded.jobs}
    assert done_id.to_wire() in succeeded_ids
    assert pending_id.to_wire() not in succeeded_ids

    pending = harness.service.list_jobs(controller_pb2.Controller.ListJobsRequest(state_filter="pending"), None)
    pending_ids = {j.job_id for j in pending.jobs}
    assert pending_id.to_wire() in pending_ids
    assert done_id.to_wire() not in pending_ids


def test_list_jobs_filter_by_name(harness: ServiceTestHarness):
    """list_jobs name_filter matches job names containing the substring."""
    harness.submit("exp-a-job")
    harness.submit("exp-b-job")
    other_id = harness.submit("other-job")

    resp = harness.service.list_jobs(controller_pb2.Controller.ListJobsRequest(name_filter="exp-"), None)
    job_ids = {j.job_id for j in resp.jobs}

    assert other_id.to_wire() not in job_ids
    assert len(job_ids) >= 2


def test_terminate_job(harness: ServiceTestHarness):
    """terminate_job transitions a running job to KILLED."""
    _ensure_workers(harness)
    job_id = harness.submit("term-me")

    harness.service.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire()), None)

    status = harness.get_job_status(job_id)
    assert status.state == job_pb2.JOB_STATE_KILLED


def test_terminate_job_skips_finished(harness: ServiceTestHarness):
    """Terminating an already-succeeded job is a no-op (no error)."""
    _ensure_workers(harness)
    job_id = harness.submit("already-done")
    harness.drive_job_to_completion(job_id)

    status = harness.get_job_status(job_id)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED

    # Terminating a finished job should not raise
    harness.service.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire()), None)
    # State should remain SUCCEEDED
    status = harness.get_job_status(job_id)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


def test_submit_rejects_name_with_slash(harness: ServiceTestHarness):
    """Job names containing '/' at the leaf are rejected."""
    # JobName.root already validates that the name segment is clean,
    # so constructing a wire name with a slash in the leaf is invalid.
    with pytest.raises(ValueError):
        JobName.root("test-user", "invalid/name")
