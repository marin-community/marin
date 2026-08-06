# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.errors import ConnectError
from iris.rpc import job_pb2


def test_root_cancellation_terminates_its_live_child_tree(journey):
    root = journey.submit("root")
    journey.settle()
    child = journey.submit_child(root, "child", tasks=2)
    journey.settle()

    journey.cancel(root)
    journey.settle()

    assert journey.job(root).state == job_pb2.JOB_STATE_KILLED
    assert journey.job(child).state == job_pb2.JOB_STATE_KILLED
    assert [task.state for task in journey.tasks(child)] == [job_pb2.TASK_STATE_KILLED] * 2


def test_finished_parent_refuses_a_new_child(journey):
    root = journey.submit("finished-parent")
    journey.settle()
    journey.succeed(root[0])
    journey.settle()

    with pytest.raises(ConnectError, match="has terminated"):
        journey.submit_child(root, "late-child")
