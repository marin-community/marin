# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Controller --dry-run mode."""

import pytest

pytestmark = pytest.mark.timeout(15)


@pytest.fixture
def dry_run_controller(make_controller):
    return make_controller(dry_run=True)


def test_dry_run_checkpoint_returns_sentinel(dry_run_controller):
    controller = dry_run_controller
    path, result = controller.begin_checkpoint()
    assert path == "dry-run"
    assert result.job_count == 0
    assert result.task_count == 0
    assert result.worker_count == 0
