# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.types import JobName
from iris.rpc import job_pb2

from scripts.ci.collect_perf_metrics import build_report, fetch_job_summary, fetch_job_tree, fetch_leaf_summaries


def _timestamp(epoch_ms: int) -> dict[str, int]:
    return {"epoch_ms": epoch_ms}


def _task(
    job_id: str,
    index: int,
    state: job_pb2.TaskState,
    *,
    duration_ms: int,
    peak_mb: int = 0,
    exit_code: int = 0,
    error: str = "",
) -> job_pb2.TaskStatus:
    return job_pb2.TaskStatus(
        task_id=f"{job_id}/{index}",
        state=state,
        exit_code=exit_code,
        error=error,
        started_at=_timestamp(1_000),
        finished_at=_timestamp(1_000 + duration_ms),
        resource_usage={"memory_peak_mb": peak_mb},
    )


class FakeIrisClient:
    def __init__(self) -> None:
        self.jobs = {
            "/u/run": job_pb2.JobStatus(
                job_id="/u/run",
                state=job_pb2.JOB_STATE_FAILED,
                started_at=_timestamp(1_000),
                finished_at=_timestamp(11_000),
                preemption_count=1,
                failure_count=0,
                task_count=1,
                completed_count=1,
                task_state_counts={"failed": 1},
                has_children=True,
            ),
            "/u/run/zephyr-normalize-work": job_pb2.JobStatus(
                job_id="/u/run/zephyr-normalize-work",
                state=job_pb2.JOB_STATE_FAILED,
                started_at=_timestamp(2_000),
                finished_at=_timestamp(5_000),
                preemption_count=2,
                failure_count=1,
                task_count=1,
                completed_count=1,
                task_state_counts={"failed": 1},
                has_children=False,
                parent_job_id="/u/run",
            ),
        }
        self.tasks = {
            "/u/run": [_task("/u/run", 0, job_pb2.TASK_STATE_FAILED, duration_ms=10_000, exit_code=1)],
            "/u/run/zephyr-normalize-work": [
                _task(
                    "/u/run/zephyr-normalize-work",
                    0,
                    job_pb2.TASK_STATE_FAILED,
                    duration_ms=3_000,
                    peak_mb=2048,
                    exit_code=137,
                    error="OOM",
                )
            ],
        }

    def list_jobs(self, *, prefix: str) -> list[job_pb2.JobStatus]:
        return [job for job_id, job in self.jobs.items() if job_id.startswith(prefix)]

    def status(self, job_name: JobName) -> job_pb2.JobStatus:
        return self.jobs[job_name.to_wire()]

    def list_tasks(self, job_name: JobName) -> list[job_pb2.TaskStatus]:
        return self.tasks[job_name.to_wire()]


def test_perf_report_uses_iris_client_tree_and_leaf_summaries() -> None:
    client = FakeIrisClient()

    summary = fetch_job_summary(client, "/u/run")
    job_tree = fetch_job_tree(client, "/u/run")
    assert job_tree is not None
    leaf_summaries = fetch_leaf_summaries(client, job_tree)

    report = build_report(
        job_id="/u/run",
        summary=summary,
        job_tree=job_tree,
        leaf_summaries=leaf_summaries,
        status={"status": "failed", "marin_prefix": "gs://bucket/prefix"},
        workflow_env={},
    )

    assert report.wall_seconds_total == 10.0
    assert report.stage_wall_seconds["normalize"] == 3.0
    assert report.preemption_count == 3
    assert report.failure_count == 1
    assert report.task_state_counts == {"failed": 2}
    assert report.tree_job_count == 2
    assert report.peak_worker_memory_mb == 2048
    assert report.ooms == 1
