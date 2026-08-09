# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for job lifecycle operations through the RPC service layer.

These tests exercise the legacy controller-service API parameterized across
both GCP and K8s providers via the ServiceTestHarness.
"""

import pytest
from connectrpc.errors import ConnectError
from iris.rpc import controller_pb2

from .conftest import ServiceTestHarness


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


@pytest.mark.parametrize(
    "query_kwargs",
    [
        pytest.param({"name_filter": "exp-"}, id="name_filter_substring"),
        # job_id_prefix needs the user segment because the match is anchored
        # against the full wire-form job_id.
        pytest.param({"job_id_prefix": "/test-user/exp-"}, id="job_id_prefix_anchored"),
    ],
)
def test_list_jobs_filter_includes_only_matching(harness: ServiceTestHarness, query_kwargs):
    """Both ListJobs filter fields exclude non-matching jobs."""
    harness.submit("exp-a-job")
    harness.submit("exp-b-job")
    other_id = harness.submit("other-job")

    resp = harness.service.list_jobs(
        controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(**query_kwargs)),
        None,
    )
    job_ids = {j.job_id for j in resp.jobs}

    assert other_id.to_wire() not in job_ids
    assert len(job_ids) >= 2
