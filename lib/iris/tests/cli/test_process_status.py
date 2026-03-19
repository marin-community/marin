# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.process_status — profile target resolution."""

import pytest

from iris.cli.process_status import _resolve_profile_target


def test_resolve_profile_target_controller():
    target, label = _resolve_profile_target(worker=None, task=None)
    assert target == "/system/process"
    assert label == "Controller"


def test_resolve_profile_target_worker():
    target, label = _resolve_profile_target(worker="worker-abc", task=None)
    assert target == "/system/worker/worker-abc"
    assert "worker-abc" in label


def test_resolve_profile_target_task():
    target, label = _resolve_profile_target(worker=None, task="/alice/my-job/0")
    assert target == "/alice/my-job/0"
    assert "/alice/my-job/0" in label


def test_resolve_profile_target_task_with_attempt():
    target, label = _resolve_profile_target(worker=None, task="/alice/my-job/0:2")
    assert target == "/alice/my-job/0:2"
    assert "/alice/my-job/0:2" in label


def test_resolve_profile_target_mutual_exclusion():
    import click

    with pytest.raises(click.UsageError, match="mutually exclusive"):
        _resolve_profile_target(worker="worker-abc", task="/alice/job/0")
