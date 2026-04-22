# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
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
lifecycle orchestration without SSH, Docker, or cloud API calls.

FakeWorkerHandle controls health check outcomes via run_command() responses:
when ``healthy=True``, the curl health check returns success immediately;
when ``healthy=False``, it returns failure. A short health_check_timeout
ensures unhealthy tests complete quickly without real polling delays.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import pytest

from iris.cluster.controller.vm_lifecycle import (
    _build_controller_vm_config,
    start_controller,
    stop_controller,
)
from iris.cluster.providers.types import (
    CloudWorkerState,
    CommandResult,
    SliceHandle,
    WorkerStatus,
)
from iris.rpc import config_pb2
from rigging.timing import Duration

# Short timeout so unhealthy health checks fail fast in tests
_TEST_HEALTH_CHECK_TIMEOUT = 0.5

# ============================================================================
# Fakes
# ============================================================================


class FakeWorkerHandle:
    """Fake StandaloneWorkerHandle for testing lifecycle orchestration.

    Tracks calls to terminate, bootstrap, set_labels, set_metadata so tests
    can verify the lifecycle functions interact with workers correctly.

    When ``healthy=False``, run_command returns non-zero exit codes, causing
    the real ``wait_healthy`` to report the VM as unhealthy.
    """

    def __init__(
        self,
        vm_id: str = "fake-vm-1",
        internal_address: str = "10.0.0.1",
        wait_for_connection_result: bool = True,
        healthy: bool = True,
    ):
        self._vm_id = vm_id
        self._internal_address = internal_address
        self._wait_for_connection_result = wait_for_connection_result
        self._healthy = healthy
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

    @property
    def bootstrap_log(self) -> str:
        return ""

    @property
    def worker_id(self) -> str:
        return self._vm_id

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
        if self._healthy:
            return CommandResult(returncode=0, stdout="ok", stderr="")
        return CommandResult(returncode=1, stdout="", stderr="connection refused")

    def bootstrap(self, script: str) -> None:
        self.bootstrap_calls.append(script)

    def reboot(self) -> None:
        pass

    def terminate(self, *, wait: bool = False) -> None:
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

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

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

    def list_all_slices(self) -> list[SliceHandle]:
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
    config.controller.image = "ghcr.io/test/iris:latest"
    config.platform.label_prefix = label_prefix
    config.defaults.ssh.user = "root"
    return config


@pytest.fixture
def config() -> config_pb2.IrisClusterConfig:
    return _make_config()


# ============================================================================
# start_controller
# ============================================================================


def test_start_controller_fresh(config):
    """No existing controller VM -- creates a new one, bootstraps, health checks, labels."""
    new_vm = FakeWorkerHandle(vm_id="ctrl-new", internal_address="10.0.0.5")
    platform = FakePlatform(existing_vms=[], vm_to_create=new_vm)

    address, vm = start_controller(platform, config)

    assert address == "http://10.0.0.5:10000"
    assert vm is new_vm
    assert len(new_vm.bootstrap_calls) == 1
    assert new_vm.labels == {"iris-test-controller": "true"}
    assert "iris-test-controller-address" in new_vm.metadata
    assert not new_vm.terminated


def test_start_controller_reuses_healthy_existing(config):
    """Existing labeled VM is healthy -- reuses it without creating a new one."""
    existing = FakeWorkerHandle(vm_id="ctrl-existing", internal_address="10.0.0.2")
    platform = FakePlatform(existing_vms=[existing])

    address, vm = start_controller(platform, config)

    assert address == "http://10.0.0.2:10000"
    assert vm is existing
    assert len(platform.created_vms) == 0
    assert not existing.terminated
    assert len(existing.bootstrap_calls) == 0


def test_start_controller_replaces_unhealthy_existing(config):
    """Existing VM is unhealthy -- terminates it and creates a fresh one."""
    unhealthy = FakeWorkerHandle(vm_id="ctrl-sick", internal_address="10.0.0.2", healthy=False)
    replacement = FakeWorkerHandle(vm_id="ctrl-new", internal_address="10.0.0.6")
    platform = FakePlatform(existing_vms=[unhealthy], vm_to_create=replacement)

    address, vm = start_controller(platform, config, health_check_timeout=_TEST_HEALTH_CHECK_TIMEOUT)

    assert unhealthy.terminated
    assert vm is replacement
    assert address == "http://10.0.0.6:10000"
    assert len(replacement.bootstrap_calls) == 1


def test_start_controller_fresh_terminates_existing_and_bootstraps_with_flag(config):
    """fresh=True -- existing healthy VM is terminated, new VM gets --fresh in bootstrap."""
    existing = FakeWorkerHandle(vm_id="ctrl-existing", internal_address="10.0.0.2")
    replacement = FakeWorkerHandle(vm_id="ctrl-new", internal_address="10.0.0.7")
    platform = FakePlatform(existing_vms=[existing], vm_to_create=replacement)

    address, vm = start_controller(platform, config, fresh=True)

    assert existing.terminated
    assert vm is replacement
    assert address == "http://10.0.0.7:10000"
    assert len(replacement.bootstrap_calls) == 1
    assert "--fresh" in replacement.bootstrap_calls[0]


def test_start_controller_fresh_new_vm_bootstrap_contains_flag(config):
    """fresh=True with no existing VM still threads --fresh into the bootstrap script."""
    new_vm = FakeWorkerHandle(vm_id="ctrl-new", internal_address="10.0.0.8")
    platform = FakePlatform(existing_vms=[], vm_to_create=new_vm)

    start_controller(platform, config, fresh=True)

    assert len(new_vm.bootstrap_calls) == 1
    assert "--fresh" in new_vm.bootstrap_calls[0]


def test_start_controller_without_fresh_omits_flag(config):
    """fresh=False (default) must not inject --fresh into the bootstrap script."""
    new_vm = FakeWorkerHandle(vm_id="ctrl-new", internal_address="10.0.0.9")
    platform = FakePlatform(existing_vms=[], vm_to_create=new_vm)

    start_controller(platform, config)

    assert len(new_vm.bootstrap_calls) == 1
    assert "--fresh" not in new_vm.bootstrap_calls[0]


def test_start_controller_connection_timeout_terminates_vm(config):
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


def test_stop_controller_found(config):
    """Finds controller VM and terminates it."""
    existing = FakeWorkerHandle(vm_id="ctrl-to-stop")
    platform = FakePlatform(existing_vms=[existing])

    stop_controller(platform, config)

    assert existing.terminated


def test_stop_controller_not_found_does_not_raise(config):
    """No controller VM found -- no error raised."""
    platform = FakePlatform(existing_vms=[])

    # Should not raise
    stop_controller(platform, config)


def test_start_controller_duplicate_vms_raises(config):
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


def test_gcp_controller_vm_config_defaults_to_500gb_disk():
    """GCP controller VM defaults to 500GB disk (sized for log-store retention)."""
    config = config_pb2.IrisClusterConfig()
    config.platform.label_prefix = "test"
    config.controller.gcp.zone = "us-central1-a"

    vm_config = _build_controller_vm_config(config)

    assert vm_config.gcp.boot_disk_size_gb == 500
