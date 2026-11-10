# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from iris.client.workload import Device, DeviceKind, JobStatus, ResourceRequest
from iris.client.workload_codec import job_status_from_proto
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from scripts.ops.old_iris_jobs import MS_PER_DAY, OldJob, find_old_jobs, report

NOW_MS = 1_800_000_000_000
CUTOFF_MS = NOW_MS - 7 * MS_PER_DAY
TPU_TASK = ResourceRequest(
    cpu_millicores=8000, memory_bytes=2**34, disk_bytes=0, device=Device(DeviceKind.TPU, "v6e-8", 8)
)
CPU_TASK = ResourceRequest(cpu_millicores=500, memory_bytes=2**32, disk_bytes=0)
GPU_TASK = ResourceRequest(cpu_millicores=0, memory_bytes=0, disk_bytes=0, device=Device(DeviceKind.GPU, "H100", 8))


def _running_job(
    job_id: str, *, started_days_ago: float, task_states: dict[str, int], cluster: str = "local"
) -> JobStatus:
    proto = job_pb2.JobStatus(
        job_id=job_id,
        state=job_pb2.JOB_STATE_RUNNING,
        task_state_counts=task_states,
        cluster=cluster,
    )
    proto.started_at.epoch_ms = NOW_MS - int(started_days_ago * MS_PER_DAY)
    return job_status_from_proto(proto)


def _old_job(name: str, started_days_ago: float, *, cluster: str = "marin") -> OldJob:
    return OldJob(
        cluster, JobName.from_wire(f"/alice/{name}"), NOW_MS - int(started_days_ago * MS_PER_DAY), {"tpu=v5p-8": 1}
    )


def test_find_old_jobs_counts_active_tasks_of_each_tree_on_the_cluster_that_runs_it():
    jobs = [
        _running_job("/alice/train", started_days_ago=10, task_states={"running": 2, "assigned": 1, "pending": 5}),
        _running_job("/alice/train/eval", started_days_ago=2, task_states={"building": 1}),
        _running_job("/alice/train/done", started_days_ago=9, task_states={"succeeded": 3}),
        _running_job("/bob/fresh", started_days_ago=2, task_states={"running": 1}),
        _running_job("/carol/preempted", started_days_ago=10, task_states={"pending": 2}),
        _running_job("/dave/gang", started_days_ago=9, task_states={"running": 8}, cluster="cw-rno2a"),
    ]
    # The lookup fails for any job without an active task, so it also checks those are skipped.
    requests = {
        JobName.from_wire("/alice/train"): TPU_TASK,
        JobName.from_wire("/alice/train/eval"): CPU_TASK,
        JobName.from_wire("/dave/gang"): GPU_TASK,
    }

    old_jobs = find_old_jobs("marin", jobs, CUTOFF_MS, requests.__getitem__)

    assert sorted(old_jobs, key=lambda job: job.key) == [
        OldJob("cw-rno2a", JobName.from_wire("/dave/gang"), NOW_MS - 9 * MS_PER_DAY, {"gpu=8xH100": 8}),
        OldJob(
            "marin",
            JobName.from_wire("/alice/train"),
            NOW_MS - 10 * MS_PER_DAY,
            {"tpu=v6e-8": 3, "cpu=0.5 mem=4GiB": 1},
        ),
        OldJob("marin", JobName.from_wire("/carol/preempted"), NOW_MS - 10 * MS_PER_DAY, {}),
    ]


def test_report_lists_each_old_job_once_and_defers_jobs_beyond_the_message_size():
    old_jobs = [_old_job(f"job-{index:03d}", started_days_ago=8 + index) for index in range(80)]

    first_message, first_reported = report(old_jobs, set(), "marin", NOW_MS, 7)
    second_message, second_reported = report(old_jobs, first_reported, "marin", NOW_MS, 7)
    third_message, third_reported = report(old_jobs, second_reported, "marin", NOW_MS, 7)

    oldest_first = sorted(old_jobs, key=lambda job: job.started_at_ms)
    posted_first = [job for job in oldest_first if job.key in first_reported]
    assert 0 < len(posted_first) < len(old_jobs)
    # Loom rejects a Slack post above 4,000 bytes.
    assert len(first_message.encode()) <= 4_000
    assert posted_first == oldest_first[: len(posted_first)]
    assert all(str(job.root) in first_message for job in posted_first)
    assert all(str(job.root) in second_message for job in old_jobs if job.key not in first_reported)
    assert second_reported == {job.key for job in old_jobs}
    assert third_message == ""
    assert third_reported == second_reported


def test_report_lists_a_job_again_after_it_stops_running():
    kept, finished, resubmitted = _old_job("kept", 9), _old_job("finished", 9), _old_job("resubmitted", 30)
    _, first_reported = report([kept, finished, resubmitted], set(), "marin", NOW_MS, 7)
    _, second_reported = report([kept], first_reported, "marin", NOW_MS, 7)

    message, third_reported = report([kept, resubmitted], second_reported, "marin", NOW_MS, 7)

    assert second_reported == {kept.key}
    assert third_reported == {kept.key, resubmitted.key}
    assert str(resubmitted.root) in message
    assert str(kept.root) not in message


def test_report_lists_a_job_once_across_a_preemption_that_leaves_every_task_pending():
    job = _old_job("preemptible", 9)
    waiting = replace(job, occupied={})
    never_posted = replace(_old_job("never-posted", 9), occupied={})

    first_message, first_reported = report([job, never_posted], set(), "marin", NOW_MS, 7)
    waiting_message, waiting_reported = report([waiting, never_posted], first_reported, "marin", NOW_MS, 7)
    resumed_message, _ = report([job, never_posted], waiting_reported, "marin", NOW_MS, 7)

    assert str(job.root) in first_message
    assert str(never_posted.root) not in first_message
    assert waiting_message == ""
    assert waiting_reported == {job.key}
    assert resumed_message == ""


def test_report_tells_owners_to_cancel_a_federated_job_through_the_scanned_cluster():
    federated = _old_job("federated", 9, cluster="cw-rno2a")

    message, _ = report([federated], set(), "marin", NOW_MS, 7)

    assert "iris --cluster=marin job cancel" in message
    assert "--cluster=cw-rno2a" not in message
