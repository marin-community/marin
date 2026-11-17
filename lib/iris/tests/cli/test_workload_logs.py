# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for workload log commands."""

from typing import cast

import pytest
from click.testing import CliRunner
from finelog.rpc import logging_pb2
from iris.cli.attempt import attempt_logs
from iris.cli.job import logs as job_logs
from iris.cli.task import task_logs
from iris.client import IrisClient
from iris.cluster.client import ClusterClient
from iris.cluster.types import JobName
from iris.rpc import job_pb2


class _FinishedWorkloadTransport:
    def get_job_states(self, job_ids: list[JobName]):
        return {job_id.to_wire(): job_pb2.JOB_STATE_SUCCEEDED for job_id in job_ids}

    def get_task_status(self, task_name: JobName, *, deadline=None):
        task = job_pb2.TaskStatus(
            task_id=task_name.to_wire(),
            state=job_pb2.TASK_STATE_SUCCEEDED,
            current_attempt_id=0,
        )
        task.attempts.add(attempt_id=0, attempt_uid="attempt-0", state=job_pb2.TASK_STATE_SUCCEEDED)
        return task

    def fetch_logs(self, source: str, *, cursor: int = 0, **_kwargs):
        if cursor:
            return logging_pb2.FetchLogsResponse(cursor=1)
        return logging_pb2.FetchLogsResponse(
            entries=[
                logging_pb2.LogEntry(
                    timestamp=logging_pb2.Timestamp(epoch_ms=1_000),
                    source="stdout",
                    data="training started",
                    attempt_id=0,
                    key="/alice/train/0:0",
                    seq=1,
                )
            ],
            cursor=1,
        )

    def shutdown(self, wait: bool = True) -> None:
        pass


@pytest.mark.parametrize(
    ("command", "target", "client_factory"),
    [
        (job_logs, "/alice/train", "iris.cli.job._remote_client"),
        (task_logs, "/alice/train/0", "iris.cli.task.iris_client_for_ctx"),
        (attempt_logs, "/alice/train/0:0", "iris.cli.attempt.iris_client_for_ctx"),
    ],
)
def test_workload_log_commands_follow_until_the_selected_resource_finishes(monkeypatch, command, target, client_factory):
    client = IrisClient(cast(ClusterClient, _FinishedWorkloadTransport()))
    monkeypatch.setattr(client_factory, lambda *_args, **_kwargs: client)

    result = CliRunner().invoke(
        command,
        [target, "--follow"],
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert result.output.endswith("| training started\n")
