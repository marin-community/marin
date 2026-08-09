# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import MagicMock

from iris.client import IrisClient
from iris.cluster.types import JobName
from iris.resources.execution import Entrypoint, EnvironmentSpec, GpuDevice, ResourceSpec
from iris.resources.identity import JobIdentity, ResourceKey, ResourceKind
from iris.resources.job import JobSummary
from iris.resources.source import Page
from iris.resources.state import JobState
from rigging.timing import Timestamp


def _job(job_id: str, uid: str) -> JobSummary:
    return JobSummary(
        identity=JobIdentity(ResourceKey("test", ResourceKind.JOB, job_id), uid),
        owner_id="alice",
        parent=None,
        state=JobState.RUNNING,
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


def test_high_level_submit_applies_accelerator_cpu_floor_without_changing_direct_specs() -> None:
    cluster = MagicMock()
    cluster.submit_job.return_value = JobIdentity(
        ResourceKey("test", ResourceKind.JOB, "/alice/train"),
        "job-uid",
    )
    client = IrisClient(cluster)
    requested = ResourceSpec(cpu=0.5, device=GpuDevice("H100"))

    client.submit(
        Entrypoint.from_command("python", "train.py"),
        "train",
        requested,
        environment=EnvironmentSpec(setup_scripts=()),
        user="alice",
    )

    submitted = cluster.submit_job.call_args.args[0]
    assert submitted.resources.cpu == 4
    assert requested.cpu == 0.5

    client.submit(
        Entrypoint.from_command("python", "train.py"),
        "large-train",
        ResourceSpec(cpu=6, device=GpuDevice("H100")),
        environment=EnvironmentSpec(setup_scripts=()),
        user="alice",
    )
    assert cluster.submit_job.call_args.args[0].resources.cpu == 6

    direct = replace(submitted, resources=requested)
    client.submit_job(direct)
    assert cluster.submit_job.call_args.args[0].resources.cpu == 0.5
