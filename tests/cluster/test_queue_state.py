# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from iris.cluster.types import JobName
from iris.resources.state import TaskState

from tests.cluster.conftest import _task_left_queue


@pytest.mark.parametrize(
    "state,started",
    [(TaskState.PENDING, False), (TaskState.BUILDING, False), (TaskState.ASSIGNED, False), (TaskState.RUNNING, True)],
)
def test_typed_task_states_preserve_the_queue_runtime_boundary(state, started):
    client = SimpleNamespace(task_status=lambda task: SimpleNamespace(state=state))
    assert _task_left_queue(client, JobName.from_wire("/test/job/0")) is started
