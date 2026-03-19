# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.process_status — profile target resolution."""

from iris.cli.process_status import _resolve_profile_target


def test_resolve_profile_target_controller():
    target, label = _resolve_profile_target(None)
    assert target == "/system/process"
    assert label == "Controller"


def test_resolve_profile_target_worker_id():
    target, label = _resolve_profile_target("worker-abc")
    assert target == "/system/worker/worker-abc"
    assert "worker-abc" in label


def test_resolve_profile_target_task_path():
    target, label = _resolve_profile_target("/alice/my-job/0")
    assert target == "/alice/my-job/0"
    assert "/alice/my-job/0" in label


def test_resolve_profile_target_task_path_with_attempt():
    target, label = _resolve_profile_target("/alice/my-job/0:2")
    assert target == "/alice/my-job/0:2"
    assert "/alice/my-job/0:2" in label


def test_resolve_profile_target_system_path():
    # Full system paths (e.g. /system/worker/abc) pass through directly
    target, _label = _resolve_profile_target("/system/worker/abc")
    assert target == "/system/worker/abc"
