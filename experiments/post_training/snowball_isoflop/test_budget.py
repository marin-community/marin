# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from iris.client.workload import BuildMetrics, TaskStatus
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from rigging.timing import Timestamp

from experiments.post_training.snowball_isoflop.budget import allocated_seconds


def test_unadmitted_queue_time_is_excluded_but_assigned_setup_is_bounded():
    task = TaskStatus(
        task_id=JobName.from_wire("/benfeuer/snowball-isoflop-budget-test/probe/0"),
        state=TaskState.RUNNING,
        worker_id="",
        worker_address="",
        exit_code=0,
        error_message="",
        submitted_at=Timestamp.from_seconds(0),
        started_at=Timestamp.from_seconds(700),
        finished_at=None,
        ports={},
        resource_usage=None,
        build_metrics=None,
        current_attempt_number=0,
        attempts=(),
        pending_reason="",
        can_be_scheduled=True,
        container_id="",
        backend_id="",
        execution_cluster_id="cw-rno2a",
        status_message="",
    )
    now = Timestamp.from_seconds(1000)
    assert allocated_seconds(task, now) == 300
    finished = replace(task, state=TaskState.SUCCEEDED, finished_at=Timestamp.from_seconds(900))
    assert allocated_seconds(finished, now) == 200
    queued = replace(task, state=TaskState.BUILDING, started_at=None)
    assert allocated_seconds(queued, now) == 0
    assigned = replace(queued, worker_id="allocated-worker")
    assert allocated_seconds(assigned, now) == 0
    building = replace(
        assigned,
        build_metrics=BuildMetrics(Timestamp.from_seconds(500), None, False, "test-image"),
    )
    assert allocated_seconds(building, now) == 500
