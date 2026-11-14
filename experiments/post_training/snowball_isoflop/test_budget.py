# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from iris.client.workload import TaskStatus
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from rigging.timing import Timestamp

from experiments.post_training.snowball_isoflop.budget import allocated_seconds


def test_missing_worker_build_timestamp_does_not_omit_reserved_setup_time():
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
    job_started = Timestamp.from_seconds(100)

    assert allocated_seconds(task, now, job_started) == 900
    finished = replace(task, state=TaskState.SUCCEEDED, finished_at=Timestamp.from_seconds(900))
    assert allocated_seconds(finished, now, job_started) == 800
