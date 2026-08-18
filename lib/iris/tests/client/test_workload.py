# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public workload handles over the deployed ControllerService transport."""

from typing import cast

from finelog.rpc import logging_pb2
from iris.client import Attempt, IrisClient
from iris.cluster.client import ClusterClient
from iris.cluster.types import JobName, TaskAttempt
from iris.resources.state import TaskState
from iris.rpc import controller_pb2, job_pb2


class _WorkloadTransport:
    def __init__(self) -> None:
        self.description = controller_pb2.Controller.GetTaskStatusResponse()
        task = self.description.task
        task.task_id = "/alice/train/0"
        task.state = job_pb2.TASK_STATE_RUNNING
        task.current_attempt_id = 3
        for number, state in ((2, job_pb2.TASK_STATE_PREEMPTED), (3, job_pb2.TASK_STATE_RUNNING)):
            attempt = task.attempts.add()
            attempt.attempt_id = number
            attempt.attempt_uid = f"attempt-{number}"
            attempt.state = state

        self.log_requests: list[tuple[str, int, str]] = []
        self.actions: list[tuple[list[str], job_pb2.TaskState, str]] = []

    def get_task_status(self, _task_name: JobName):
        return self.description.task

    def get_task_description(self, _task_name: JobName):
        return self.description

    def fetch_logs(self, source: str, *, match_scope: int, substring: str, **_kwargs):
        self.log_requests.append((source, match_scope, substring))
        entry = logging_pb2.LogEntry(
            timestamp=logging_pb2.Timestamp(epoch_ms=2000),
            source="stdout",
            data="ready marker",
            attempt_id=2,
            key="/alice/train/0:2",
        )
        return logging_pb2.FetchLogsResponse(entries=[entry])

    def kick_tasks(self, targets: list[str], desired_state: job_pb2.TaskState, reason: str):
        self.actions.append((targets, desired_state, reason))
        return [
            controller_pb2.Controller.KickResult(
                target=target,
                task_id=target.partition(":")[0],
                queued=not target.endswith(":2"),
                detail="attempt is no longer current" if target.endswith(":2") else "",
            )
            for target in targets
        ]


def _client(transport: _WorkloadTransport) -> IrisClient:
    return IrisClient(cast(ClusterClient, transport))


def test_task_and_attempt_handles_select_current_and_historical_execution():
    transport = _WorkloadTransport()
    task = _client(transport).job(JobName.from_wire("/alice/train")).task(0)

    assert task.status().state is TaskState.RUNNING
    assert [attempt.attempt_number for attempt in task.attempts()] == [2, 3]
    current_attempt = cast(Attempt, task.current_attempt())
    assert current_attempt.ref == TaskAttempt.from_wire("/alice/train/0:3")

    entries = task.attempt(2).logs(substring="marker")

    assert [entry.data for entry in entries] == ["ready marker"]
    assert entries[0].task_id == JobName.from_wire("/alice/train/0")
    assert transport.log_requests == [
        ("/alice/train/0:2", logging_pb2.MATCH_SCOPE_EXACT, "marker"),
    ]


def test_task_actions_preserve_current_or_numbered_target_and_acceptance():
    transport = _WorkloadTransport()
    client = _client(transport)
    task = client.task(JobName.from_wire("/alice/train/0"))

    current = task.preempt(reason="rebalance")
    historical = client.attempt(TaskAttempt.from_wire("/alice/train/0:2")).fail(reason="bad output")

    assert current.accepted is True
    assert current.task_id == JobName.from_wire("/alice/train/0")
    assert historical.accepted is False
    assert historical.message == "attempt is no longer current"
    assert transport.actions == [
        (["/alice/train/0"], job_pb2.TASK_STATE_PREEMPTED, "rebalance"),
        (["/alice/train/0:2"], job_pb2.TASK_STATE_FAILED, "bad output"),
    ]
