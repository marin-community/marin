# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.resources.errors import ResourcePreconditionFailed
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

    assert journey.job(root).summary.state == job_pb2.JOB_STATE_KILLED
    assert journey.job(child).summary.state == job_pb2.JOB_STATE_KILLED
    assert [task.state for task in journey.tasks(child)] == [job_pb2.TASK_STATE_KILLED] * 2
    assert journey.job(finished_child).summary.state == job_pb2.JOB_STATE_SUCCEEDED
    assert journey.job(unrelated).summary.state == job_pb2.JOB_STATE_RUNNING


def test_parent_failure_terminates_its_live_child(journey):
    root = journey.submit("failed-root")
    journey.settle()
    child = journey.submit_child(root, "child")
    journey.settle()

    journey.fail(root[0])
    journey.settle()

    assert journey.job(root).summary.state == job_pb2.JOB_STATE_FAILED
    assert journey.job(child).summary.state == job_pb2.JOB_STATE_KILLED


def test_child_cancellation_preserves_its_parent_and_sibling(journey):
    root = journey.submit("selective-cancel")
    journey.settle()
    child = journey.submit_child(root, "target")
    sibling = journey.submit_child(root, "sibling")
    journey.settle()

    journey.cancel(child)
    journey.settle()

    assert journey.job(child).summary.state == job_pb2.JOB_STATE_KILLED
    assert journey.job(sibling).summary.state == job_pb2.JOB_STATE_RUNNING
    assert journey.job(root).summary.state == job_pb2.JOB_STATE_RUNNING


def test_finished_parent_refuses_a_new_child(journey):
    root = journey.submit("finished-parent")
    journey.settle()
    journey.succeed(root[0])
    journey.settle()

    with pytest.raises(ResourcePreconditionFailed, match="has terminated"):
        journey.submit_child(root, "late-child")
