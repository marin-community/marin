# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cli.bug_report import _gather, format_bug_report
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2, logging_pb2


class _FakeBugReportClient:
    def __init__(self, root_job_id: str, task_id: str):
        self._root_job_id = root_job_id
        self._task_id = task_id

    def get_job_status(self, request):
        assert request.job_id == self._root_job_id
        launch_request = cluster_pb2.Controller.LaunchJobRequest(name=self._root_job_id)
        launch_request.entrypoint.run_command.argv[:] = ["python", "train.py"]
        return cluster_pb2.Controller.GetJobStatusResponse(
            job=cluster_pb2.JobStatus(
                job_id=self._root_job_id,
                name=self._root_job_id,
                state=cluster_pb2.JOB_STATE_RUNNING,
                task_count=1,
                completed_count=0,
                failure_count=0,
                preemption_count=0,
            ),
            request=launch_request,
        )

    def list_tasks(self, request):
        assert request.job_id == self._root_job_id
        return cluster_pb2.Controller.ListTasksResponse(
            tasks=[
                cluster_pb2.TaskStatus(
                    task_id=self._task_id,
                    state=cluster_pb2.TASK_STATE_RUNNING,
                    worker_id="worker-0",
                    worker_address="10.0.0.1:10001",
                )
            ]
        )

    def list_workers(self, _request):
        return cluster_pb2.Controller.ListWorkersResponse()

    def get_task_logs(self, request):
        if request.id == self._task_id:
            return cluster_pb2.Controller.GetTaskLogsResponse(
                task_logs=[
                    cluster_pb2.Controller.TaskLogBatch(
                        task_id=self._task_id,
                        logs=[logging_pb2.LogEntry(source="stdout", data="still running")],
                    )
                ]
            )

        assert request.id == self._root_job_id
        assert request.include_children
        return cluster_pb2.Controller.GetTaskLogsResponse(
            child_job_statuses=[
                cluster_pb2.JobStatus(
                    job_id=f"{self._root_job_id}/child-failed",
                    state=cluster_pb2.JOB_STATE_FAILED,
                    exit_code=1,
                    error="Exit code: 1. stderr: boom",
                ),
                cluster_pb2.JobStatus(
                    job_id=f"{self._root_job_id}/child-running",
                    state=cluster_pb2.JOB_STATE_RUNNING,
                ),
            ]
        )


def test_bug_report_surfaces_descendant_job_failures():
    root_job_id = "/test-user/root-job"
    task_id = f"{root_job_id}/0"
    report = _gather(_FakeBugReportClient(root_job_id, task_id), JobName.from_wire(root_job_id), tail=20)

    assert report.state_name == "running"
    assert report.error_summary == "1 descendant job(s) failed"
    assert [job.job_id for job in report.descendant_jobs] == [
        f"{root_job_id}/child-failed",
        f"{root_job_id}/child-running",
    ]

    markdown = format_bug_report(report)

    assert "## Descendant Jobs" in markdown
    assert "Descendant Failures | 1 |" in markdown
    assert "child-failed" in markdown
    assert "Exit code: 1. stderr: boom" in markdown
