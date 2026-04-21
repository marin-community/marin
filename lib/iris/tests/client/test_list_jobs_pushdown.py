# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify IrisClient.list_jobs pushes filters into the server JobQuery.

Server-side filtering matters: without it, a simple
``iris job list --state running`` triggers a full-table paginated scan
(offset=0, 500, 1000, ...) which is expensive on large deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from iris.client import IrisClient
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2


@dataclass
class _RecordingClusterClient:
    """Captures the JobQuery passed into list_jobs; returns canned jobs."""

    jobs: list[job_pb2.JobStatus] = field(default_factory=list)
    captured_queries: list[controller_pb2.Controller.JobQuery] = field(default_factory=list)

    def list_jobs(
        self,
        *,
        query: controller_pb2.Controller.JobQuery | None = None,
        page_size: int = 500,
    ) -> list[job_pb2.JobStatus]:
        del page_size
        captured = controller_pb2.Controller.JobQuery()
        if query is not None:
            captured.CopyFrom(query)
        self.captured_queries.append(captured)
        return list(self.jobs)

    def shutdown(self, wait: bool = True) -> None:
        del wait

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


def _make_status(job_id: str, state: job_pb2.JobState) -> job_pb2.JobStatus:
    status = job_pb2.JobStatus()
    status.job_id = job_id
    status.state = state
    return status


@pytest.fixture
def stub_client() -> tuple[IrisClient, _RecordingClusterClient]:
    stub = _RecordingClusterClient()
    client = IrisClient(cluster=stub)
    return client, stub


def test_list_jobs_no_filter_sends_empty_query(stub_client):
    client, stub = stub_client
    stub.jobs = [_make_status("/u/a", job_pb2.JOB_STATE_RUNNING)]

    client.list_jobs()

    assert len(stub.captured_queries) == 1
    q = stub.captured_queries[0]
    assert q.state_filter == ""
    assert q.name_filter == ""


def test_list_jobs_state_is_pushed_down(stub_client):
    client, stub = stub_client
    stub.jobs = [_make_status("/u/a", job_pb2.JOB_STATE_RUNNING)]

    client.list_jobs(state=job_pb2.JOB_STATE_RUNNING)

    assert stub.captured_queries[0].state_filter == "running"


def test_list_jobs_prefix_is_pushed_down(stub_client):
    client, stub = stub_client
    prefix = JobName.root("alice", "exp")
    stub.jobs = [_make_status(prefix.to_wire() + "-1", job_pb2.JOB_STATE_PENDING)]

    client.list_jobs(prefix=prefix)

    assert stub.captured_queries[0].name_filter == prefix.to_wire()


def test_list_jobs_reanchors_prefix_client_side(stub_client):
    """name_filter is a substring; client must still enforce startswith.

    `/bob/alice/exp-oops` contains `/alice/exp` as a substring so it
    slips through the server-side LIKE '%...%' filter. The client must
    drop it because it is not a true prefix of `/alice/exp`.
    """
    client, stub = stub_client
    prefix = JobName.root("alice", "exp")
    wire = prefix.to_wire()
    stub.jobs = [
        _make_status(wire, job_pb2.JOB_STATE_RUNNING),
        _make_status(wire + "-child", job_pb2.JOB_STATE_RUNNING),
        _make_status("/bob/alice/exp-oops", job_pb2.JOB_STATE_RUNNING),
    ]

    result = client.list_jobs(prefix=prefix)

    assert {j.job_id for j in result} == {wire, wire + "-child"}
