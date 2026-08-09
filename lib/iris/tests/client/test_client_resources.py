# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from iris.client import IrisClient
from iris.cluster.resources.identity import JobIdentity, ResourceKey, ResourceKind
from iris.cluster.resources.job import JobSummary
from iris.cluster.resources.source import Page
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp


def _job(job_id: str, uid: str) -> JobSummary:
    return JobSummary(
        identity=JobIdentity(ResourceKey("test", ResourceKind.JOB, job_id), uid),
        owner_id="alice",
        parent=None,
        state=job_pb2.JOB_STATE_RUNNING,
        execution_cluster_id="test",
        backend_id="default",
        num_tasks=1,
        submitted_at=Timestamp.from_ms(1),
        started_at=None,
        finished_at=None,
        error_message="",
        pending_reason="",
    )


def test_current_job_finds_exact_job_beyond_first_prefix_page() -> None:
    exact = _job("/alice/train", "exact-uid")
    cluster = MagicMock()
    cluster.list_jobs.side_effect = (
        Page(tuple(_job(f"/alice/train/child-{index}", f"child-{index}") for index in range(500)), "next", ()),
        Page((exact,), None, ()),
    )
    client = IrisClient(cluster)

    job = client.current_job(JobName.from_wire("/alice/train"))

    assert job.identity == exact.identity
