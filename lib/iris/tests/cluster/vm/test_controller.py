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

"""Tests for controller lifecycle management."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris.cluster.vm.config import ControllerVmConfig, IrisClusterConfig, load_config
from iris.cluster.vm.controller import (
    CONTROLLER_CONTAINER_NAME,
    GcpController,
    ManualController,
    create_controller,
)


@pytest.fixture
def ssh_bootstrap_config() -> IrisClusterConfig:
    """Config for SSH bootstrap mode."""
    return IrisClusterConfig(
        provider_type="manual",
        controller_vm=ControllerVmConfig(
            enabled=False,
            host="10.0.0.100",
            image="gcr.io/project/iris-controller:latest",
            port=10000,
        ),
        ssh_user="ubuntu",
        ssh_private_key="/home/ubuntu/.ssh/id_rsa",
    )


@pytest.fixture
def gcp_config() -> IrisClusterConfig:
    """Config for GCP controller VM."""
    return IrisClusterConfig(
        provider_type="gcp",
        project_id="my-project",
        zone="us-central1-a",
        controller_vm=ControllerVmConfig(
            enabled=True,
            image="gcr.io/project/iris-controller:latest",
            port=10000,
        ),
    )


def test_manual_controller_requires_host():
    """ManualController requires controller_vm.host."""
    config = IrisClusterConfig(
        provider_type="manual",
        controller_vm=ControllerVmConfig(enabled=False),
    )
    with pytest.raises(RuntimeError, match=r"controller_vm\.host is required"):
        ManualController(config)


def test_manual_controller_start_requires_image():
    """start() requires image to be configured."""
    config = IrisClusterConfig(
        provider_type="manual",
        controller_vm=ControllerVmConfig(
            host="10.0.0.100",
            image="",
        ),
    )
    controller = ManualController(config)
    with pytest.raises(RuntimeError, match="image required"):
        controller.start()


@patch("iris.cluster.vm.controller.check_health")
@patch("iris.cluster.vm.controller.run_streaming_with_retry")
@patch("iris.cluster.vm.controller.DirectSshConnection")
def test_manual_controller_start_runs_bootstrap(
    mock_conn_cls: MagicMock,
    mock_run_streaming: MagicMock,
    mock_health: MagicMock,
    ssh_bootstrap_config: IrisClusterConfig,
):
    """start() SSHs into host and runs bootstrap script."""
    mock_conn = MagicMock()
    mock_conn_cls.return_value = mock_conn
    mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_health.return_value = True

    controller = ManualController(ssh_bootstrap_config)
    result = controller.start()

    assert result == "http://10.0.0.100:10000"
    mock_conn_cls.assert_called_once_with(
        host="10.0.0.100",
        user="ubuntu",
        key_file="/home/ubuntu/.ssh/id_rsa",
        connect_timeout=30,
    )
    mock_run_streaming.assert_called_once()
    # Bootstrap script should reference docker and controller container
    call_args = mock_run_streaming.call_args
    command = call_args[0][1]
    assert "docker" in command
    assert "iris-controller" in command


@patch("iris.cluster.vm.controller.check_health")
@patch("iris.cluster.vm.controller.run_streaming_with_retry")
@patch("iris.cluster.vm.controller.DirectSshConnection")
def test_manual_controller_stop_runs_stop_script(
    mock_conn_cls: MagicMock,
    mock_run_streaming: MagicMock,
    mock_health: MagicMock,
    ssh_bootstrap_config: IrisClusterConfig,
):
    """stop() runs docker stop via SSH after bootstrap."""
    mock_conn = MagicMock()
    mock_conn.run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_conn_cls.return_value = mock_conn
    mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_health.return_value = True

    controller = ManualController(ssh_bootstrap_config)
    controller.start()
    controller.stop()

    # Verify stop command was run
    assert mock_conn.run.called
    stop_command = mock_conn.run.call_args[0][0]
    assert "docker stop" in stop_command
    assert CONTROLLER_CONTAINER_NAME in stop_command


@patch("iris.cluster.vm.controller.DirectSshConnection")
def test_manual_controller_stop_skipped_if_not_bootstrapped(
    mock_conn_cls: MagicMock,
    ssh_bootstrap_config: IrisClusterConfig,
):
    """stop() is no-op if we didn't bootstrap."""
    mock_conn = MagicMock()
    mock_conn_cls.return_value = mock_conn

    controller = ManualController(ssh_bootstrap_config)
    controller.stop()  # Don't call start() first

    mock_conn.run.assert_not_called()


@patch("iris.cluster.vm.controller.check_health")
@patch("iris.cluster.vm.controller.run_streaming_with_retry")
@patch("iris.cluster.vm.controller.DirectSshConnection")
def test_manual_controller_reload_calls_start(
    mock_conn_cls: MagicMock,
    mock_run_streaming: MagicMock,
    mock_health: MagicMock,
    ssh_bootstrap_config: IrisClusterConfig,
):
    """reload() delegates to start() for ManualController."""
    mock_conn = MagicMock()
    mock_conn_cls.return_value = mock_conn
    mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_health.return_value = True

    controller = ManualController(ssh_bootstrap_config)
    result = controller.reload()

    assert result == "http://10.0.0.100:10000"
    # Should have run bootstrap (via start)
    mock_run_streaming.assert_called_once()


def test_create_controller_returns_gcp_when_enabled(gcp_config: IrisClusterConfig):
    """create_controller returns GcpController when VM enabled."""
    controller = create_controller(gcp_config)
    assert isinstance(controller, GcpController)


def test_create_controller_returns_manual_for_manual_provider(ssh_bootstrap_config: IrisClusterConfig):
    """create_controller returns ManualController for manual provider."""
    controller = create_controller(ssh_bootstrap_config)
    assert isinstance(controller, ManualController)


class TestConfigParsing:
    """Tests for config parsing with controller VM settings."""

    def test_load_config_with_controller_host(self, tmp_path: Path):
        """Config with controller.vm.host is parsed correctly."""
        config_content = """\
provider:
  type: manual

docker:
  image: gcr.io/project/iris-worker:latest

controller:
  vm:
    enabled: false
    host: 10.0.0.100
    image: gcr.io/project/iris-controller:latest
    port: 10000

auth:
  ssh_user: ubuntu
  ssh_private_key: ~/.ssh/id_rsa

manual_hosts:
  - 10.0.0.1
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.controller_vm.host == "10.0.0.100"
        assert config.controller_vm.image == "gcr.io/project/iris-controller:latest"
        assert config.controller_vm.port == 10000
        assert config.ssh_user == "ubuntu"
