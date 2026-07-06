# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Controller --dry-run mode."""

from unittest.mock import MagicMock

import pytest
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.conftest import (
    autoscale_once,
    make_job_request,
    make_worker_metadata,
    query_tasks_for_job,
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
    state = ControllerTestState(
        controller._db,
        # The single backend owns the liveness tracker and attrs projection now;
        # register workers into them so the controller's schedule path sees them.
        health=controller.provider.health,
    )

    register_worker(state, "w1", "w1:8080", make_worker_metadata())
    req = make_job_request(name="dry-job", cpu=1, replicas=1)
    submit_job(state, "dry-job", req)

    controller._run_scheduling()

    tasks = query_tasks_for_job(state, JobName.root("test-user", "dry-job"))
    assert len(tasks) == 1
    assert tasks[0].state == job_pb2.TASK_STATE_PENDING


def test_dry_run_autoscaler_skipped_entirely(dry_run_controller):
    controller = dry_run_controller
    controller._representative_backend.autoscale = MagicMock()

    # In dry-run the control tick short-circuits to the schedule-only path, so
    # the autoscale phase never reaches the backend even when forced.
    autoscale_once(controller)

    controller._representative_backend.autoscale.assert_not_called()


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
