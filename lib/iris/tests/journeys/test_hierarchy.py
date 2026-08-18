# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.errors import ConnectError
from iris.rpc import job_pb2


def test_root_cancellation_terminates_only_live_descendants(journey):
    root = journey.submit("root")
    journey.settle()
    finished_child = journey.submit_child(root, "finished")
    journey.settle()
    journey.succeed(finished_child[0])
    journey.settle()
    child = journey.submit_child(root, "child", tasks=2)
    journey.settle()
    unrelated = journey.submit("unrelated")
    journey.settle()

    journey.cancel(root)
    journey.settle()

    assert journey.job(root).state == job_pb2.JOB_STATE_KILLED
    assert journey.job(child).state == job_pb2.JOB_STATE_KILLED
    assert [task.state for task in journey.tasks(child)] == [job_pb2.TASK_STATE_KILLED] * 2
    assert journey.job(finished_child).state == job_pb2.JOB_STATE_SUCCEEDED
    assert journey.job(unrelated).state == job_pb2.JOB_STATE_RUNNING


def test_parent_failure_terminates_child_with_cumulative_failure_cause(journey):
    root = journey.submit("failed-root")
    journey.settle()
    child = journey.submit_child(root, "child")
    journey.settle()

    journey.fail(root[0])
    journey.settle()

    root_status = journey.job(root)
    child_status = journey.job(child)
    assert root_status.state == job_pb2.JOB_STATE_FAILED
    assert "max_task_failures" in root_status.error
    assert "failures=1" in root_status.error
    assert "limit=0" in root_status.error
    assert "application failure" in root_status.error
    assert child_status.state == job_pb2.JOB_STATE_KILLED
    assert child_status.error == root_status.error


def test_child_cancellation_preserves_its_parent_and_sibling(journey):
    root = journey.submit("selective-cancel")
    journey.settle()
    child = journey.submit_child(root, "target")
    sibling = journey.submit_child(root, "sibling")
    journey.settle()

    journey.cancel(child)
    journey.settle()

    assert journey.job(child).state == job_pb2.JOB_STATE_KILLED
    assert journey.job(sibling).state == job_pb2.JOB_STATE_RUNNING
    assert journey.job(root).state == job_pb2.JOB_STATE_RUNNING


def test_finished_parent_refuses_a_new_child(journey):
    root = journey.submit("finished-parent")
    journey.settle()
    journey.succeed(root[0])
    journey.settle()

    with pytest.raises(ConnectError, match="has terminated"):
        journey.submit_child(root, "late-child")
