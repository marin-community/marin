# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for controller lifecycle functions (start, stop).

Uses fake Platform and StandaloneWorkerHandle implementations to exercise the
lifecycle orchestration without SSH, Docker, or cloud API calls. The
wait_healthy function is patched since it does real SSH polling internally.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from unittest.mock import patch

import pytest

from iris.cluster.controller.vm_lifecycle import (
    start_controller,
    stop_controller,
)
from iris.cluster.platform.base import (
    CloudWorkerState,
    CommandResult,
    SliceHandle,
    WorkerStatus,
)
from iris.rpc import config_pb2
from iris.time_utils import Duration

# ============================================================================
# Fakes
# ============================================================================


class FakeWorkerHandle:
    """Fake StandaloneWorkerHandle for testing lifecycle orchestration.

    Tracks calls to terminate, bootstrap, set_labels, set_metadata so tests
    can verify the lifecycle functions interact with workers correctly.
    """

    def __init__(
        self,
        vm_id: str = "fake-vm-1",
        internal_address: str = "10.0.0.1",
        wait_for_connection_result: bool = True,
    ):
        self._vm_id = vm_id
        self._internal_address = internal_address
        self._wait_for_connection_result = wait_for_connection_result
        self.terminated = False
        self.bootstrap_calls: list[str] = []
        self.labels: dict[str, str] = {}
        self.metadata: dict[str, str] = {}

    @property
    def vm_id(self) -> str:
        return self._vm_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        return self._wait_for_connection_result

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        return CommandResult(returncode=0, stdout="", stderr="")

    def bootstrap(self, script: str) -> None:
        self.bootstrap_calls.append(script)

    def reboot(self) -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True

    def set_labels(self, labels: dict[str, str]) -> None:
        self.labels.update(labels)

    def set_metadata(self, metadata: dict[str, str]) -> None:
        self.metadata.update(metadata)


class FakePlatform:
    """Fake Platform that returns pre-configured VMs from list_vms and create_vm."""

    def __init__(
        self,
        existing_vms: list[FakeWorkerHandle] | None = None,
        vm_to_create: FakeWorkerHandle | None = None,
    ):
        self._existing_vms = existing_vms or []
        self._vm_to_create = vm_to_create or FakeWorkerHandle()
        self.created_vms: list[FakeWorkerHandle] = []

    def create_vm(self, config: config_pb2.VmConfig) -> FakeWorkerHandle:
        self.created_vms.append(self._vm_to_create)
        return self._vm_to_create

    def create_slice(self, config: config_pb2.SliceConfig) -> SliceHandle:
        raise NotImplementedError

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[SliceHandle]:
        return []

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[SliceHandle]:
        return []

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[FakeWorkerHandle]:
        return list(self._existing_vms)

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        return nullcontext(address)

    def shutdown(self) -> None:
        pass


# ============================================================================
# Fixtures
# ============================================================================


def _make_config(
    label_prefix: str = "test",
    host: str = "10.0.0.1",
    port: int = 10000,
) -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    config.controller.manual.host = host
    config.controller.manual.port = port
    config.platform.label_prefix = label_prefix
    config.defaults.ssh.user = "root"
    return config


@pytest.fixture
def config() -> config_pb2.IrisClusterConfig:
    return _make_config()


LIFECYCLE_MODULE = "iris.cluster.controller.vm_lifecycle"


# ============================================================================
# start_controller
# ============================================================================


@patch(f"{LIFECYCLE_MODULE}.build_controller_bootstrap_script_from_config", return_value="#!/bin/bash\necho ok")
@patch(f"{LIFECYCLE_MODULE}.wait_healthy", return_value=True)
def test_start_controller_fresh(mock_wait_healthy, mock_bootstrap_script, config):
    """No existing controller VM -- creates a new one, bootstraps, health checks, labels."""
    new_vm = FakeWorkerHandle(vm_id="ctrl-new", internal_address="10.0.0.5")
    platform = FakePlatform(existing_vms=[], vm_to_create=new_vm)

    address, vm = start_controller(platform, config)

    assert address == "http://10.0.0.5:10000"
    assert vm is new_vm
    assert len(new_vm.bootstrap_calls) == 1
    assert new_vm.labels == {"test-controller": "true"}
    assert "test-controller-address" in new_vm.metadata
    assert not new_vm.terminated


@patch(f"{LIFECYCLE_MODULE}.build_controller_bootstrap_script_from_config", return_value="#!/bin/bash\necho ok")
@patch(f"{LIFECYCLE_MODULE}.wait_healthy", return_value=True)
def test_start_controller_reuses_healthy_existing(mock_wait_healthy, mock_bootstrap_script, config):
    """Existing labeled VM is healthy -- reuses it without creating a new one."""
    existing = FakeWorkerHandle(vm_id="ctrl-existing", internal_address="10.0.0.2")
    platform = FakePlatform(existing_vms=[existing])

    address, vm = start_controller(platform, config)

    assert address == "http://10.0.0.2:10000"
    assert vm is existing
    assert len(platform.created_vms) == 0
    assert not existing.terminated
    # Should not have bootstrapped the existing healthy VM
    assert len(existing.bootstrap_calls) == 0


@patch(f"{LIFECYCLE_MODULE}.build_controller_bootstrap_script_from_config", return_value="#!/bin/bash\necho ok")
@patch(f"{LIFECYCLE_MODULE}.wait_healthy")
def test_start_controller_replaces_unhealthy_existing(mock_wait_healthy, mock_bootstrap_script, config):
    """Existing VM is unhealthy -- terminates it and creates a fresh one."""
    unhealthy = FakeWorkerHandle(vm_id="ctrl-sick", internal_address="10.0.0.2")
    replacement = FakeWorkerHandle(vm_id="ctrl-new", internal_address="10.0.0.6")
    platform = FakePlatform(existing_vms=[unhealthy], vm_to_create=replacement)

    # First call checks existing (unhealthy), second call checks new VM (healthy)
    mock_wait_healthy.side_effect = [False, True]

    address, vm = start_controller(platform, config)

    assert unhealthy.terminated
    assert vm is replacement
    assert address == "http://10.0.0.6:10000"
    assert len(replacement.bootstrap_calls) == 1


@patch(f"{LIFECYCLE_MODULE}.build_controller_bootstrap_script_from_config", return_value="#!/bin/bash\necho ok")
@patch(f"{LIFECYCLE_MODULE}.wait_healthy", return_value=True)
def test_start_controller_connection_timeout_terminates_vm(mock_wait_healthy, mock_bootstrap_script, config):
    """VM created but wait_for_connection fails -- terminates the VM and raises."""
    unreachable = FakeWorkerHandle(
        vm_id="ctrl-unreachable",
        internal_address="10.0.0.9",
        wait_for_connection_result=False,
    )
    platform = FakePlatform(existing_vms=[], vm_to_create=unreachable)

    with pytest.raises(RuntimeError, match="did not become reachable"):
        start_controller(platform, config)

    assert unreachable.terminated


# ============================================================================
# stop_controller
# ============================================================================


@patch(f"{LIFECYCLE_MODULE}.wait_healthy", return_value=True)
def test_stop_controller_found(mock_wait_healthy, config):
    """Finds controller VM and terminates it."""
    existing = FakeWorkerHandle(vm_id="ctrl-to-stop")
    platform = FakePlatform(existing_vms=[existing])

    stop_controller(platform, config)

    assert existing.terminated


@patch(f"{LIFECYCLE_MODULE}.wait_healthy", return_value=True)
def test_stop_controller_not_found(mock_wait_healthy, config):
    """No controller VM found -- no error raised."""
    platform = FakePlatform(existing_vms=[])

    # Should not raise
    stop_controller(platform, config)


@patch(f"{LIFECYCLE_MODULE}.build_controller_bootstrap_script_from_config", return_value="#!/bin/bash\necho ok")
@patch(f"{LIFECYCLE_MODULE}.wait_healthy", return_value=True)
def test_start_controller_duplicate_vms_raises(mock_wait_healthy, mock_bootstrap_script, config):
    """Multiple controller VMs found -- raises RuntimeError listing duplicates."""
    vm1 = FakeWorkerHandle(vm_id="ctrl-dup-1", internal_address="10.0.0.1")
    vm2 = FakeWorkerHandle(vm_id="ctrl-dup-2", internal_address="10.0.0.2")
    platform = FakePlatform(existing_vms=[vm1, vm2])

    with pytest.raises(RuntimeError, match="Multiple controller VMs found"):
        start_controller(platform, config)


def test_stop_controller_duplicate_vms_raises(config):
    """stop_controller with multiple matching VMs raises RuntimeError."""
    vm1 = FakeWorkerHandle(vm_id="ctrl-dup-1", internal_address="10.0.0.1")
    vm2 = FakeWorkerHandle(vm_id="ctrl-dup-2", internal_address="10.0.0.2")
    platform = FakePlatform(existing_vms=[vm1, vm2])

    with pytest.raises(RuntimeError, match="Multiple controller VMs found"):
        stop_controller(platform, config)
