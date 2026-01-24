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

"""Tests for Iris CLI commands.

Tests are organized by command group: build, cluster, controller, slice, vm, and autoscaler.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from iris.cli import iris
from iris.rpc import vm_pb2
from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


def _create_test_autoscaler(scale_group_name: str = "test-group"):
    """Create a test Autoscaler with FakeVmManager."""
    from iris.cluster.vm.autoscaler import Autoscaler, AutoscalerConfig
    from iris.cluster.vm.managed_vm import VmRegistry
    from iris.cluster.vm.scaling_group import ScalingGroup

    sg_config = vm_pb2.ScaleGroupConfig(
        name=scale_group_name,
        accelerator_type="v5p-8",
        min_slices=0,
        max_slices=10,
        zones=["us-central1-a"],
    )
    fake_manager = FakeVmManager(FakeVmManagerConfig(config=sg_config))
    scale_group = ScalingGroup(config=sg_config, vm_manager=fake_manager)

    return (
        Autoscaler(
            scale_groups={scale_group_name: scale_group},
            vm_registry=VmRegistry(),
            config=AutoscalerConfig(),
        ),
        fake_manager,
    )


@pytest.fixture
def test_autoscaler():
    """Create a test Autoscaler for CLI testing."""
    autoscaler, fake_manager = _create_test_autoscaler()
    return autoscaler, fake_manager


@pytest.fixture
def mock_config():
    """Create a mock IrisClusterConfig for CLI testing."""
    config = MagicMock()
    config.controller_address = "10.0.0.100:10000"
    config.scale_groups = {"test-group": MagicMock()}
    return config


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config = {
        "provider": {"type": "manual"},
        "manual_hosts": ["10.0.0.1", "10.0.0.2"],
        "auth": {"ssh_user": "root"},
        "docker": {"image": "test-image:latest", "worker_port": 10001},
        "controller": {"address": "10.0.0.100:10000"},
        "scale_groups": {
            "manual": {"accelerator_type": "cpu", "min_slices": 0, "max_slices": 10},
        },
    }
    config_path = tmp_path / "vm-config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


# =============================================================================
# Build Command Tests
# =============================================================================

# Build tests removed per AGENTS.md - they only test log messages/output text format


# =============================================================================
# Cluster Command Tests
# =============================================================================

# Cluster command tests removed per AGENTS.md - they only test log messages/output text format


# =============================================================================
# Controller Command Tests
# =============================================================================

# Controller command tests removed per AGENTS.md - they only test log messages/output text format


# =============================================================================
# VM Status Command Tests (via controller)
# =============================================================================

# VM status command tests removed per AGENTS.md - they only test log messages/output text format


def test_vm_status_via_config(cli_runner: CliRunner, config_file: Path, test_autoscaler, mock_config):
    """VM status command works with --config."""
    autoscaler, fake_manager = test_autoscaler
    group = autoscaler.get_group("test-group")
    slice_obj = group.scale_up()
    fake_manager.tick()

    with patch("iris.cli._load_autoscaler", return_value=(autoscaler, mock_config)):
        result = cli_runner.invoke(iris, ["cluster", "--config", str(config_file), "vm", "status"])
        assert result.exit_code == 0
        assert slice_obj.slice_id in result.output


# =============================================================================
# VM Logs Command Tests
# =============================================================================

# VM logs display tests removed per AGENTS.md - they only test log messages/output text format


def test_vm_logs_command_with_tail(cli_runner: CliRunner):
    """VM logs command passes tail parameter."""
    with patch("iris.cli._get_vm_logs") as mock_get_logs:
        mock_get_logs.return_value = ("last line", "vm-123", vm_pb2.VM_STATE_READY)

        result = cli_runner.invoke(
            iris,
            ["cluster", "vm", "--controller-url", "http://localhost:10000", "logs", "vm-123", "--tail", "10"],
        )

        assert result.exit_code == 0
        mock_get_logs.assert_called_once_with("http://localhost:10000", "vm-123", 10)


# =============================================================================
# Autoscaler Status Command Tests (via controller)
# =============================================================================

# Autoscaler status command tests removed per AGENTS.md - they only test log messages/output text format


# =============================================================================
# Slice Command Tests
# =============================================================================

# Slice create/list tests removed per AGENTS.md - they only test log messages/output text format


def test_slice_list_with_data(cli_runner: CliRunner, config_file: Path, test_autoscaler, mock_config):
    """Slice list shows slices in table format."""
    autoscaler, fake_manager = test_autoscaler
    group = autoscaler.get_group("test-group")
    slice_obj = group.scale_up()
    fake_manager.tick()

    with patch("iris.cli._load_autoscaler", return_value=(autoscaler, mock_config)):
        result = cli_runner.invoke(iris, ["cluster", "--config", str(config_file), "slice", "list"])
        assert result.exit_code == 0
        assert slice_obj.slice_id in result.output


def test_slice_get_success(cli_runner: CliRunner, config_file: Path, test_autoscaler, mock_config):
    """Slice get shows detailed slice info."""
    autoscaler, fake_manager = test_autoscaler
    group = autoscaler.get_group("test-group")
    slice_obj = group.scale_up()
    fake_manager.tick()

    with patch("iris.cli._load_autoscaler", return_value=(autoscaler, mock_config)):
        result = cli_runner.invoke(iris, ["cluster", "--config", str(config_file), "slice", "get", slice_obj.slice_id])
        assert result.exit_code == 0
        assert f"Slice: {slice_obj.slice_id}" in result.output


# Slice get/terminate tests removed per AGENTS.md - they only test log messages/output text format


def test_slice_terminate_multiple(cli_runner: CliRunner, config_file: Path, test_autoscaler, mock_config):
    """Slice terminate accepts multiple slice IDs."""
    autoscaler, _ = test_autoscaler
    group = autoscaler.get_group("test-group")
    slice1 = group.scale_up()
    slice2 = group.scale_up()

    with patch("iris.cli._load_autoscaler", return_value=(autoscaler, mock_config)):
        result = cli_runner.invoke(
            iris, ["cluster", "--config", str(config_file), "slice", "terminate", slice1.slice_id, slice2.slice_id]
        )
        assert result.exit_code == 0
        assert slice1.slice_id in result.output
        assert slice2.slice_id in result.output
