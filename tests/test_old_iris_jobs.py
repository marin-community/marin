# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from iris.client.workload import Device, DeviceKind, JobStatus, ResourceRequest
from iris.client.workload_codec import job_status_from_proto
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from scripts.ops.old_iris_jobs import MS_PER_DAY, OldJob, occupied_resources, old_trees, report

NOW_MS = 1_800_000_000_000
CUTOFF_MS = NOW_MS - 7 * MS_PER_DAY


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


def _old_job(name: str, started_days_ago: float, *, scanned_cluster: str = "marin", cluster: str = "marin") -> OldJob:
    return OldJob(
        scanned_cluster,
        cluster,
        JobName.from_wire(f"/alice/{name}"),
        NOW_MS - int(started_days_ago * MS_PER_DAY),
        {"tpu=v5p-8": 1},
    )


def test_old_trees_group_children_under_their_root_on_the_cluster_that_runs_them():
    jobs = [
        _running_job("/alice/train", started_days_ago=10, task_states={"running": 1}),
        _running_job("/alice/train/eval", started_days_ago=2, task_states={"running": 4}),
        _running_job("/bob/fresh", started_days_ago=2, task_states={"running": 1}),
        _running_job("/carol/waiting", started_days_ago=10, task_states={"pending": 2}),
        _running_job("/dave/gang", started_days_ago=9, task_states={"running": 1}, cluster="cw-rno2a"),
        _running_job("/dave/gang/workers", started_days_ago=9, task_states={"running": 8}, cluster="cw-rno2a"),
    ]

    trees = old_trees("marin", jobs, CUTOFF_MS)

    assert {(tree.cluster, tree.root, tree.started_at_ms): {job.job_id for job in tree.jobs} for tree in trees} == {
        ("marin", JobName.from_wire("/alice/train"), NOW_MS - 10 * MS_PER_DAY): {
            JobName.from_wire("/alice/train"),
            JobName.from_wire("/alice/train/eval"),
        },
        ("marin", JobName.from_wire("/carol/waiting"), NOW_MS - 10 * MS_PER_DAY): {
            JobName.from_wire("/carol/waiting"),
        },
        ("cw-rno2a", JobName.from_wire("/dave/gang"), NOW_MS - 9 * MS_PER_DAY): {
            JobName.from_wire("/dave/gang"),
            JobName.from_wire("/dave/gang/workers"),
        },
    }


def test_occupied_resources_count_only_active_tasks_per_task_request():
    jobs = [
        _running_job("/alice/train", started_days_ago=10, task_states={"running": 2, "assigned": 1, "pending": 5}),
        _running_job("/alice/train/coordinator", started_days_ago=10, task_states={"building": 1}),
        _running_job("/alice/train/done", started_days_ago=10, task_states={"succeeded": 3}),
    ]
    requests = {
        JobName.from_wire("/alice/train"): ResourceRequest(
            cpu_millicores=8000, memory_bytes=2**34, disk_bytes=0, device=Device(DeviceKind.TPU, "v6e-8", 8)
        ),
        JobName.from_wire("/alice/train/coordinator"): ResourceRequest(
            cpu_millicores=500, memory_bytes=2**32, disk_bytes=0
        ),
    }

    assert occupied_resources(jobs, requests.__getitem__) == {"tpu=v6e-8": 3, "cpu=0.5 mem=4GiB": 1}


def test_report_lists_each_old_job_once_and_defers_jobs_beyond_the_message_size():
    old_jobs = [_old_job(f"job-{index:03d}", started_days_ago=8 + index) for index in range(80)]

    first_message, first_reported = report(old_jobs, set(), NOW_MS, 7)
    second_message, second_reported = report(old_jobs, first_reported, NOW_MS, 7)
    third_message, third_reported = report(old_jobs, second_reported, NOW_MS, 7)

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
    _, first_reported = report([kept, finished, resubmitted], set(), NOW_MS, 7)
    _, second_reported = report([kept], first_reported, NOW_MS, 7)

    message, third_reported = report([kept, resubmitted], second_reported, NOW_MS, 7)

    assert second_reported == {kept.key}
    assert third_reported == {kept.key, resubmitted.key}
    assert str(resubmitted.root) in message
    assert str(kept.root) not in message


def test_report_lists_a_job_once_across_a_preemption_that_leaves_every_task_pending():
    job = _old_job("preemptible", 9)
    waiting = replace(job, occupied={})
    fresh_waiting = replace(_old_job("never-posted", 9), occupied={})

    first_message, first_reported = report([job, fresh_waiting], set(), NOW_MS, 7)
    waiting_message, waiting_reported = report([waiting, fresh_waiting], first_reported, NOW_MS, 7)
    resumed_message, _ = report([job, fresh_waiting], waiting_reported, NOW_MS, 7)

    assert str(job.root) in first_message
    assert str(fresh_waiting.root) not in first_message
    assert waiting_message == ""
    assert waiting_reported == {job.key}
    assert resumed_message == ""


def test_report_tells_owners_to_cancel_through_the_cluster_that_found_the_job():
    federated = _old_job("federated", 9, scanned_cluster="marin", cluster="cw-rno2a")
    direct = _old_job("direct", 9, scanned_cluster="cw-us-east-02a", cluster="cw-us-east-02a")

    message, _ = report([federated, direct], set(), NOW_MS, 7)

    groups = {block.splitlines()[0]: block for block in message.split("\n\n")[1:]}
    assert str(federated.root) in groups["Cancel with `iris --cluster=marin job cancel <job>`:"]
    assert str(direct.root) in groups["Cancel with `iris --cluster=cw-us-east-02a job cancel <job>`:"]
