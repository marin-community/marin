# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster manager (stop_all, parallel termination)."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from iris.cluster.manager import _collect_terminate_targets, stop_all
from iris.rpc import config_pb2


class FakeSliceHandle:
    """Minimal slice handle for manager tests."""

    def __init__(self, slice_id: str, labels: dict[str, str] | None = None, terminate_delay: float = 0.0):
        self._slice_id = slice_id
        self._labels = labels or {}
        self._terminate_delay = terminate_delay
        self.terminated = False
        # Event that unblocks a slow terminate, so tests don't hang waiting
        # for ThreadPoolExecutor shutdown.
        self._release = threading.Event()

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return "test-zone"

    @property
    def scale_group(self) -> str:
        return self._labels.get("iris-scale-group", "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    def terminate(self) -> None:
        if self._terminate_delay > 0:
            self._release.wait(timeout=self._terminate_delay)
        self.terminated = True


class FakeManagerPlatform:
    """Minimal platform for manager tests with list_all_slices support."""

    def __init__(self, slices: list[FakeSliceHandle] | None = None):
        self._slices = slices or []

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[FakeSliceHandle]:
        results = list(self._slices)
        if labels:
            results = [s for s in results if all(s.labels.get(k) == v for k, v in labels.items())]
        return results

    def list_vms(self, zones: list[str], labels: dict[str, str] | None = None) -> list:
        return []

    def shutdown(self) -> None:
        pass


def _make_manager_config() -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    config.platform.label_prefix = "iris"
    config.platform.local.SetInParent()
    config.controller.local.port = 0
    return config


def test_collect_terminate_targets_discovers_managed_slices():
    """_collect_terminate_targets finds slices labeled {prefix}-managed=true."""
    managed = FakeSliceHandle("slice-1", labels={"iris-managed": "true"})
    unmanaged = FakeSliceHandle("slice-2", labels={})
    platform = FakeManagerPlatform(slices=[managed, unmanaged])
    config = _make_manager_config()

    targets = _collect_terminate_targets(platform, config)

    target_names = [name for name, _ in targets]
    assert "controller" in target_names
    assert "slice:slice-1" in target_names
    assert "slice:slice-2" not in target_names


def test_stop_all_terminates_all_slices():
    """stop_all discovers and terminates all managed slices."""
    slices = [FakeSliceHandle(f"slice-{i}", labels={"iris-managed": "true"}) for i in range(3)]
    platform = FakeManagerPlatform(slices=slices)
    config = _make_manager_config()

    with patch("iris.cluster.manager.IrisConfig") as mock_config_cls, patch("iris.cluster.manager.stop_controller"):
        mock_config_cls.return_value.platform.return_value = platform
        stop_all(config)

    assert all(s.terminated for s in slices)


def test_stop_all_timeout_logs_warning(caplog):
    """stop_all logs warning when termination exceeds timeout."""
    slow_slice = FakeSliceHandle("slow-slice", labels={"iris-managed": "true"}, terminate_delay=60)
    platform = FakeManagerPlatform(slices=[slow_slice])
    config = _make_manager_config()

    try:
        with (
            patch("iris.cluster.manager.IrisConfig") as mock_config_cls,
            patch("iris.cluster.manager.stop_controller"),
            patch("iris.cluster.manager.TERMINATE_TIMEOUT_SECONDS", 1),
        ):
            mock_config_cls.return_value.platform.return_value = platform
            with pytest.raises(RuntimeError, match="error"):
                stop_all(config)
    finally:
        # Unblock the thread so it can exit cleanly (shutdown(wait=False)
        # leaves it running until the fake delay completes).
        slow_slice._release.set()

    assert any("still running" in record.message for record in caplog.records)
