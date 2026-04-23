# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Controller --dry-run mode."""

from unittest.mock import MagicMock

import pytest

from iris.cluster.controller.schema import TASK_DETAIL_PROJECTION
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from tests.cluster.controller.conftest import (
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)

pytestmark = pytest.mark.timeout(15)


@pytest.fixture
def dry_run_controller(make_controller):
    return make_controller(dry_run=True)


def test_dry_run_controller_starts_and_stops(dry_run_controller):
    controller = dry_run_controller
    controller.start()
    assert controller.started
    controller.stop()


def test_dry_run_scheduling_does_not_dispatch(dry_run_controller):
    controller = dry_run_controller
    state = controller.state

    register_worker(state, "w1", "w1:8080", make_worker_metadata())
    req = make_job_request(name="dry-job", cpu=1, replicas=1)
    submit_job(state, "dry-job", req)

    controller._run_scheduling()

    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (JobName.root("test-user", "dry-job").to_wire(),)),
        )
    assert len(tasks) == 1
    assert tasks[0].state == job_pb2.TASK_STATE_PENDING


def test_dry_run_autoscaler_skipped_entirely(dry_run_controller):
    controller = dry_run_controller
    mock_autoscaler = MagicMock()
    controller._autoscaler = mock_autoscaler

    controller._run_autoscaler_once()

    mock_autoscaler.refresh.assert_not_called()
    mock_autoscaler.update.assert_not_called()


def test_dry_run_checkpoint_returns_sentinel(dry_run_controller):
    controller = dry_run_controller
    path, result = controller.begin_checkpoint()
    assert path == "dry-run"
    assert result.job_count == 0
    assert result.task_count == 0
    assert result.worker_count == 0


def test_dry_run_pruning_skipped(dry_run_controller):
    controller = dry_run_controller
    assert controller._prune_thread is None


def test_dry_run_kill_tasks_skipped(dry_run_controller):
    controller = dry_run_controller
    task_id = JobName.root("test-user", "fake-job").child("t0")
    controller.kill_tasks_on_workers({task_id})
