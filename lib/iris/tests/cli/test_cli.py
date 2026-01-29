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
from iris.cluster.vm.vm_platform import VmGroupStatus, VmSnapshot
from iris.rpc import config_pb2, vm_pb2


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


def _make_mock_slice(slice_id: str, scale_group: str = "test-group") -> MagicMock:
    """Create a mock VmGroupProtocol for CLI testing."""
    mock = MagicMock()
    mock.group_id = slice_id
    mock.slice_id = slice_id
    mock.scale_group = scale_group
    mock.created_at_ms = 1000000
    snapshots = [
        VmSnapshot(
            vm_id=f"{slice_id}-vm-0",
            state=vm_pb2.VM_STATE_READY,
            address="10.0.0.1",
            init_phase="",
            init_error="",
        )
    ]
    mock.status.return_value = VmGroupStatus(vms=snapshots)
    mock.to_proto.return_value = vm_pb2.SliceInfo(
        slice_id=slice_id,
        scale_group=scale_group,
        created_at_ms=1000000,
    )
    return mock


def _create_test_autoscaler(scale_group_name: str = "test-group"):
    """Create a test Autoscaler with mock VmManager."""
    from iris.cluster.vm.autoscaler import Autoscaler
    from iris.cluster.vm.managed_vm import VmRegistry
    from iris.cluster.vm.scaling_group import ScalingGroup

    sg_config = config_pb2.ScaleGroupConfig(
        name=scale_group_name,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
        min_slices=0,
        max_slices=10,
        zones=["us-central1-a"],
    )

    mock_manager = MagicMock()
    mock_manager.discover_vm_groups.return_value = []
    mock_manager._create_count = 0

    def create_vm_group_side_effect(_tags: dict[str, str] | None = None) -> MagicMock:
        mock_manager._create_count += 1
        return _make_mock_slice(f"mock-slice-{mock_manager._create_count}", scale_group_name)

    mock_manager.create_vm_group.side_effect = create_vm_group_side_effect
    scale_group = ScalingGroup(config=sg_config, vm_manager=mock_manager)

    return (
        Autoscaler(
            scale_groups={scale_group_name: scale_group},
            vm_registry=VmRegistry(),
            config=None,
        ),
        mock_manager,
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
    config.bootstrap.controller_address = "10.0.0.100:10000"
    config.scale_groups = {"test-group": MagicMock()}
    return config


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config = {
        "provider_type": "gcp",
        "project_id": "test-project",
        "region": "us-central1",
        "zone": "us-central1-a",
        "bootstrap": {
            "docker_image": "test-image:latest",
            "worker_port": 10001,
            "controller_address": "10.0.0.100:10000",
        },
        "ssh": {
            "user": "root",
        },
        "scale_groups": {
            "test-group": {
                "accelerator_type": "ACCELERATOR_TYPE_TPU",
                "accelerator_variant": "v5p-8",
                "min_slices": 0,
                "max_slices": 10,
                "zones": ["us-central1-a"],
            },
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


def test_slice_terminate_multiple(cli_runner: CliRunner, config_file: Path, test_autoscaler, mock_config):
    """Slice terminate removes slices from the autoscaler."""
    autoscaler, _ = test_autoscaler
    group = autoscaler.get_group("test-group")
    slice1 = group.scale_up()
    slice2 = group.scale_up()

    assert len(list(group.vm_groups())) == 2

    with patch("iris.cli._load_autoscaler", return_value=(autoscaler, mock_config)):
        result = cli_runner.invoke(
            iris, ["cluster", "--config", str(config_file), "slice", "terminate", slice1.slice_id, slice2.slice_id]
        )
        assert result.exit_code == 0

    # The stable behavior: slices are gone
    assert len(list(group.vm_groups())) == 0
