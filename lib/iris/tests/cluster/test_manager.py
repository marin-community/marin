# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster manager (stop_all, parallel termination)."""

from __future__ import annotations

import threading
from unittest.mock import patch

from iris.cluster.manager import stop_all
from iris.cluster.platform.base import default_stop_all
from iris.rpc import config_pb2


class FakeSliceHandle:
    """Minimal slice handle for manager tests."""

    def __init__(self, slice_id: str, labels: dict[str, str] | None = None, terminate_delay: float = 0.0):
        self._slice_id = slice_id
        self._labels = labels or {}
        self._terminate_delay = terminate_delay
        self.terminated = False
        # Event that unblocks a slow terminate, so tests can release the
        # daemon thread after stop_all returns.
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
        self.controller_stopped = False

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[FakeSliceHandle]:
        results = list(self._slices)
        if labels:
            results = [s for s in results if all(s.labels.get(k) == v for k, v in labels.items())]
        return results

    def list_vms(self, zones: list[str], labels: dict[str, str] | None = None) -> list:
        return []

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        self.controller_stopped = True

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        return default_stop_all(self, config, dry_run=dry_run, label_prefix=label_prefix)

    def shutdown(self) -> None:
        pass


def _make_manager_config() -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    config.platform.label_prefix = "iris"
    config.platform.local.SetInParent()
    config.controller.local.port = 0
    return config


def test_stop_all_dry_run_discovers_without_terminating():
    """stop_all(dry_run=True) returns resource names but does not terminate anything."""
    managed = FakeSliceHandle("slice-1", labels={"iris-managed": "true"})
    unmanaged = FakeSliceHandle("slice-2", labels={})
    platform = FakeManagerPlatform(slices=[managed, unmanaged])
    config = _make_manager_config()

    with patch("iris.cluster.manager.IrisConfig") as mock_config_cls:
        mock_config_cls.return_value.platform.return_value = platform
        names = stop_all(config, dry_run=True)

    assert "controller" in names
    assert "slice:slice-1" in names
    assert "slice:slice-2" not in names
    assert not managed.terminated
    assert not unmanaged.terminated
    assert not platform.controller_stopped


def test_stop_all_terminates_all_slices():
    """stop_all discovers and terminates all managed slices."""
    slices = [FakeSliceHandle(f"slice-{i}", labels={"iris-managed": "true"}) for i in range(3)]
    platform = FakeManagerPlatform(slices=slices)
    config = _make_manager_config()

    with patch("iris.cluster.manager.IrisConfig") as mock_config_cls:
        mock_config_cls.return_value.platform.return_value = platform
        stop_all(config)

    assert all(s.terminated for s in slices)
    assert platform.controller_stopped


def test_stop_all_custom_label_prefix():
    """stop_all with custom label_prefix uses that prefix for discovery."""
    managed = FakeSliceHandle("slice-a", labels={"custom-managed": "true"})
    wrong_prefix = FakeSliceHandle("slice-b", labels={"iris-managed": "true"})
    platform = FakeManagerPlatform(slices=[managed, wrong_prefix])
    config = _make_manager_config()

    with patch("iris.cluster.manager.IrisConfig") as mock_config_cls:
        mock_config_cls.return_value.platform.return_value = platform
        names = stop_all(config, label_prefix="custom")

    assert "slice:slice-a" in names
    assert "slice:slice-b" not in names
    assert managed.terminated
    assert not wrong_prefix.terminated
    assert platform.controller_stopped
