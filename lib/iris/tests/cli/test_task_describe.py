# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `task describe` / `attempt describe` builders and renders.

Pure over protos — the incident that motivated the vocabulary (#7542) turned on the
CLI never printing the pod name an attempt owned or the per-attempt terminal reason,
so these assert both are surfaced.
"""

import pytest
from iris.cli.task import (
    build_attempt_detail,
    build_task_description,
    render_attempt_detail_text,
    render_task_description_text,
)
from iris.rpc import controller_pb2, job_pb2


def _response() -> controller_pb2.Controller.GetTaskStatusResponse:
    """A failed two-attempt task whose current attempt owns pod ``iris-abcd-1``."""
    response = controller_pb2.Controller.GetTaskStatusResponse()
    task = response.task
    task.task_id = "/alice/job/0"
    task.state = job_pb2.TASK_STATE_FAILED
    task.backend_id = "default"
    task.current_attempt_id = 1
    task.container_id = "iris-abcd-1"

    # Added out of order to exercise the chronological sort.
    a1 = task.attempts.add()
    a1.attempt_id = 1
    a1.state = job_pb2.TASK_STATE_FAILED
    a1.exit_code = 137
    a1.worker_id = "worker-1"
    a1.attempt_uid = "cafebabecafebabe"

    a0 = task.attempts.add()
    a0.attempt_id = 0
    a0.state = job_pb2.TASK_STATE_FAILED
    a0.exit_code = 1
    a0.worker_id = "worker-0"
    a0.attempt_uid = "deadbeefdeadbeef"
    a0.error = "boom on attempt 0"
    a0.is_worker_failure = True

    response.root_cause_highlights.append("Bundle fetch abc failed: HTTP Error 404")
    return response


def test_build_task_description_surfaces_pod_and_sorted_chain():
    desc = build_task_description(_response())

    assert desc["task_id"] == "/alice/job/0"
    assert desc["state"] == "failed"
    assert desc["container_id"] == "iris-abcd-1"
    assert [a["attempt_id"] for a in desc["attempts"]] == [0, 1]
    assert desc["attempts"][0]["exit_code"] == 1
    assert desc["attempts"][0]["is_worker_failure"] is True
    assert desc["root_cause_highlights"] == ["Bundle fetch abc failed: HTTP Error 404"]


def test_task_description_render_shows_pod_root_cause_and_signal():
    text = render_task_description_text(build_task_description(_response()))

    assert "iris-abcd-1" in text  # the pod name job summary never printed
    assert "Root cause:" in text and "HTTP Error 404" in text
    assert "deadbeefdead" in text  # attempt uid, truncated
    assert "137 (SIGKILL)" in text  # exit code names the killing signal


def test_attempt_detail_current_carries_pod_and_root_cause():
    detail = build_attempt_detail(_response(), 1)

    assert detail["is_current"] is True
    assert detail["container_id"] == "iris-abcd-1"
    assert detail["root_cause_highlights"] == ["Bundle fetch abc failed: HTTP Error 404"]


def test_attempt_detail_past_omits_current_only_fields():
    detail = build_attempt_detail(_response(), 0)

    assert detail["is_current"] is False
    assert detail["container_id"] == ""
    assert detail["root_cause_highlights"] == []
    assert "not persisted for past attempts yet" in render_attempt_detail_text(detail)


def test_attempt_detail_missing_attempt_raises():
    with pytest.raises(ValueError, match="has no attempt 9"):
        build_attempt_detail(_response(), 9)
