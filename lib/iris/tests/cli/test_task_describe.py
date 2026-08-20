# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Task and Attempt descriptions."""

from typing import cast

from iris.cli.task import attempt_status, render_attempt_detail_text, render_task_description_text
from iris.client import IrisClient
from iris.cluster.client.protocol import ClusterClient
from iris.resources.names import JobName
from iris.rpc import controller_pb2, job_pb2

_INIT_FAILURE = "Init:Error stage-workdir: Bundle fetch abc failed: HTTP Error 404"


class _DescriptionTransport:
    def __init__(self) -> None:
        self.response = controller_pb2.Controller.GetTaskStatusResponse()

    def get_task_description(self, _task_name: JobName):
        return self.response


def _failed_task_description():
    transport = _DescriptionTransport()
    task = transport.response.task
    task.task_id = "/alice/job/0"
    task.state = job_pb2.TASK_STATE_FAILED
    task.backend_id = "default"
    task.current_attempt_id = 1
    task.container_id = "iris-abcd-1"

    current = task.attempts.add()
    current.attempt_id = 1
    current.state = job_pb2.TASK_STATE_FAILED
    current.exit_code = 137
    current.worker_id = "worker-1"
    current.attempt_uid = "cafebabecafebabe"
    current.pod_name = "iris-abcd-1"
    current.node_name = "node-b"

    previous = task.attempts.add()
    previous.attempt_id = 0
    previous.state = job_pb2.TASK_STATE_FAILED
    previous.exit_code = 1
    previous.worker_id = "worker-0"
    previous.attempt_uid = "deadbeefdeadbeef"
    previous.is_worker_failure = True
    previous.pod_name = "iris-abcd-0"
    previous.node_name = "node-a"
    previous.terminal_reason = _INIT_FAILURE

    transport.response.root_cause_highlights.append("Bundle fetch abc failed: HTTP Error 404")
    client = IrisClient(cast(ClusterClient, transport))
    return client.task(JobName.from_wire("/alice/job/0")).describe()


def test_task_description_surfaces_attempt_history_and_failure_diagnostics():
    text = render_task_description_text(_failed_task_description())

    assert "Backend object (current attempt 1): iris-abcd-1" in text
    assert text.index("deadbeefdead") < text.index("cafebabecafe")
    assert "HTTP Error 404" in text
    assert "137 (SIGKILL)" in text


def test_attempt_description_distinguishes_historical_and_current_diagnostics():
    description = _failed_task_description()

    previous = render_attempt_detail_text(description, attempt_status(description, 0))
    current = render_attempt_detail_text(description, attempt_status(description, 1))

    assert "Attempt: /alice/job/0:0" in previous
    assert "(current)" not in previous
    assert "Backend object: iris-abcd-0 on node-a" in previous
    assert _INIT_FAILURE in previous
    assert "Root cause:" not in previous

    assert "Attempt: /alice/job/0:1  (current)" in current
    assert "Root cause:" in current
    assert "HTTP Error 404" in current
