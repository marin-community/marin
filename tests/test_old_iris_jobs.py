# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from iris.client.workload import Device, DeviceKind, JobStatus, ResourceRequest
from iris.client.workload_codec import job_status_from_proto
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from scripts.ops.old_iris_jobs import (
    KILL_DECISION,
    MS_PER_DAY,
    OldJob,
    SheetRow,
    find_old_jobs,
    mark_failed_cancels,
    plan_tabs,
)

REVISIT_IN_30_DAYS = "Revisit in 30 days"
REVISIT_IN_365_DAYS = "Revisit in 365 days"
# 2027-01-15 08:00 UTC.
NOW_MS = 1_800_000_000_000
TODAY = "2027-01-15"
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


def _old_job(name: str, started_days_ago: float, *, cluster: str = "marin", task: ResourceRequest = TPU_TASK) -> OldJob:
    return OldJob(cluster, JobName.from_wire(name), NOW_MS - int(started_days_ago * MS_PER_DAY), {task: 1})


def _replan(old_jobs: list[OldJob], tabs: dict[str, list[SheetRow]], *, days_later: int = 0):
    return plan_tabs(old_jobs, tabs, NOW_MS + days_later * MS_PER_DAY)


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

    assert sorted(old_jobs, key=lambda job: (job.cluster, str(job.root))) == [
        OldJob("cw-rno2a", JobName.from_wire("/dave/gang"), NOW_MS - 9 * MS_PER_DAY, {GPU_TASK: 8}),
        OldJob("marin", JobName.from_wire("/alice/train"), NOW_MS - 10 * MS_PER_DAY, {TPU_TASK: 3, CPU_TASK: 1}),
        OldJob("marin", JobName.from_wire("/carol/preempted"), NOW_MS - 10 * MS_PER_DAY, {}),
    ]


def test_plan_lists_each_job_in_its_owner_tab_with_accelerator_jobs_before_older_cpu_jobs():
    cpu_oldest = _old_job("/alice/cpu-oldest", 40, task=CPU_TASK)
    gpu_newer = _old_job("/alice/gpu-newer", 8, task=GPU_TASK, cluster="cw-rno2a")
    tpu_older = _old_job("/alice/tpu-older", 20)
    email_owner = _old_job("/bob@example.edu/train", 9)

    tabs, cancels = _replan([cpu_oldest, gpu_newer, tpu_older, email_owner], {})

    assert cancels == []
    assert [row.job for row in tabs["alice"]] == ["/alice/tpu-older", "/alice/gpu-newer", "/alice/cpu-oldest"]
    assert tabs["bob"] == [
        SheetRow(
            cluster="marin",
            job="/bob@example.edu/train",
            started="2027-01-06",
            age_days="9.0",
            resources=tabs["bob"][0].resources,
            listed=TODAY,
        )
    ]
    assert "v6e-8" in tabs["bob"][0].resources
    assert tabs["alice"][1].cluster == "cw-rno2a"


def test_plan_cancels_a_job_marked_kill_until_it_stops_and_then_drops_its_row():
    job, other = _old_job("/alice/stale", 9), _old_job("/alice/other", 8)
    listed, _ = _replan([job, other], {})
    marked = {
        "alice": [replace(row, decision=KILL_DECISION) if row.job == "/alice/stale" else row for row in listed["alice"]]
    }

    first, first_cancels = _replan([job, other], marked)
    second, second_cancels = _replan([job, other], first, days_later=1)
    stopped, stopped_cancels = _replan([other], second, days_later=2)

    assert first_cancels == second_cancels == [job.root]
    assert first["alice"][0].status == f"Cancel requested {TODAY}"
    assert second["alice"][0].status == f"Cancel requested {TODAY}"
    assert stopped_cancels == []
    assert [row.job for row in stopped["alice"]] == ["/alice/other"]


def test_plan_cancels_again_a_job_whose_cancel_failed():
    job, other = _old_job("/alice/stale", 9), _old_job("/alice/other", 8)
    listed, _ = _replan([job, other], {})
    marked = {"alice": [replace(listed["alice"][0], decision=KILL_DECISION), listed["alice"][1]]}
    planned, cancels = _replan([job, other], marked)

    failed = mark_failed_cancels(planned, {str(cancels[0]): "permission denied"})
    retried, retried_cancels = _replan([job, other], failed, days_later=1)

    assert [row.status for row in failed["alice"]] == ["Cancel failed: permission denied", ""]
    assert retried_cancels == [job.root]
    assert retried["alice"][0].status == "Cancel requested 2027-01-16"


def test_plan_asks_again_on_the_revisit_date():
    job = _old_job("/alice/long", 9)
    listed, _ = _replan([job], {})
    marked = {"alice": [replace(listed["alice"][0], decision=REVISIT_IN_30_DAYS)]}

    kept, kept_cancels = _replan([job], marked)
    still_kept, _ = _replan([job], kept, days_later=29)
    expired, expired_cancels = _replan([job], still_kept, days_later=30)

    assert kept_cancels == expired_cancels == []
    assert (kept["alice"][0].decision, kept["alice"][0].revisit_on) == (REVISIT_IN_30_DAYS, "2027-02-14")
    assert (still_kept["alice"][0].decision, still_kept["alice"][0].revisit_on) == (REVISIT_IN_30_DAYS, "2027-02-14")
    assert (expired["alice"][0].decision, expired["alice"][0].revisit_on, expired["alice"][0].status) == ("", "", "")
    assert expired["alice"][0].listed == TODAY


def test_plan_starts_a_new_period_when_the_owner_changes_the_revisit_decision():
    job = _old_job("/alice/long", 9)
    listed, _ = _replan([job], {})
    kept, _ = _replan([job], {"alice": [replace(listed["alice"][0], decision=REVISIT_IN_30_DAYS)]})
    changed = {"alice": [replace(kept["alice"][0], decision=REVISIT_IN_365_DAYS, revisit_on="edited by the owner")]}

    extended, _ = _replan([job], changed, days_later=10)
    later, _ = _replan([job], extended, days_later=300)

    assert (extended["alice"][0].revisit_on, extended["alice"][0].status) == (
        "2028-01-25",
        "Revisit in 365 days from 2027-01-25",
    )
    assert later["alice"][0] == replace(extended["alice"][0], age_days=later["alice"][0].age_days)


def test_plan_flags_an_unknown_decision_without_a_cancel():
    job = _old_job("/alice/long", 9)
    listed, _ = _replan([job], {})

    tabs, cancels = _replan([job], {"alice": [replace(listed["alice"][0], decision="kill it")]})

    assert cancels == []
    assert (tabs["alice"][0].decision, tabs["alice"][0].status) == ("kill it", "Unknown decision")


def test_plan_keeps_a_row_across_a_preemption_and_empties_the_tab_when_every_job_stops():
    job = _old_job("/alice/preemptible", 9)
    never_listed = replace(_old_job("/carol/never-listed", 9), occupied={})
    listed, _ = _replan([job, never_listed], {})
    decided = {"alice": [replace(listed["alice"][0], decision=REVISIT_IN_30_DAYS)]}

    waiting, _ = _replan([replace(job, occupied={}), never_listed], decided)
    stopped, _ = _replan([never_listed], waiting)

    assert "carol" not in listed
    assert [(row.job, row.decision, row.resources) for row in waiting["alice"]] == [
        ("/alice/preemptible", REVISIT_IN_30_DAYS, "")
    ]
    assert stopped == {"alice": []}


def test_sheet_row_reads_a_row_whose_trailing_empty_cells_the_api_omits():
    row = SheetRow("marin", "/alice/long", "2027-01-06", "9.0", "1x (cpu=1)", TODAY)

    assert SheetRow.from_cells(row.cells()[:6]) == row
