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

from iris.cluster.vm.config import config_to_dict, load_config
from iris.cluster.vm.controller import (
    CONTROLLER_CONTAINER_NAME,
    GcpController,
    ManualController,
    create_controller,
)
from iris.rpc import vm_pb2


@pytest.fixture
def ssh_bootstrap_config() -> vm_pb2.IrisClusterConfig:
    """Config for SSH bootstrap mode."""
    return vm_pb2.IrisClusterConfig(
        provider_type="manual",
        controller_vm=vm_pb2.ControllerVmConfig(
            manual=vm_pb2.ManualControllerConfig(
                host="10.0.0.100",
                image="gcr.io/project/iris-controller:latest",
                port=10000,
            ),
        ),
        ssh_user="ubuntu",
        ssh_private_key="/home/ubuntu/.ssh/id_rsa",
    )


@pytest.fixture
def gcp_config() -> vm_pb2.IrisClusterConfig:
    """Config for GCP controller VM."""
    return vm_pb2.IrisClusterConfig(
        provider_type="gcp",
        project_id="my-project",
        zone="us-central1-a",
        controller_vm=vm_pb2.ControllerVmConfig(
            gcp=vm_pb2.GcpControllerConfig(
                image="gcr.io/project/iris-controller:latest",
                port=10000,
            ),
        ),
    )


def test_manual_controller_requires_host():
    """ManualController requires controller_vm.manual.host."""
    config = vm_pb2.IrisClusterConfig(
        provider_type="manual",
        controller_vm=vm_pb2.ControllerVmConfig(
            manual=vm_pb2.ManualControllerConfig(host=""),
        ),
    )
    with pytest.raises(RuntimeError, match=r"controller_vm\.manual\.host is required"):
        ManualController(config)


def test_manual_controller_start_requires_image():
    """start() requires image to be configured."""
    config = vm_pb2.IrisClusterConfig(
        provider_type="manual",
        controller_vm=vm_pb2.ControllerVmConfig(
            manual=vm_pb2.ManualControllerConfig(
                host="10.0.0.100",
                image="",
            ),
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
    ssh_bootstrap_config: vm_pb2.IrisClusterConfig,
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
    ssh_bootstrap_config: vm_pb2.IrisClusterConfig,
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
    ssh_bootstrap_config: vm_pb2.IrisClusterConfig,
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
    ssh_bootstrap_config: vm_pb2.IrisClusterConfig,
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


def test_create_controller_returns_gcp_when_enabled(gcp_config: vm_pb2.IrisClusterConfig):
    """create_controller returns GcpController when VM enabled."""
    controller = create_controller(gcp_config)
    assert isinstance(controller, GcpController)


def test_create_controller_returns_manual_for_manual_provider(ssh_bootstrap_config: vm_pb2.IrisClusterConfig):
    """create_controller returns ManualController for manual provider."""
    controller = create_controller(ssh_bootstrap_config)
    assert isinstance(controller, ManualController)


def test_create_controller_raises_on_missing_config():
    """create_controller raises ValueError when no oneof is set."""
    config = vm_pb2.IrisClusterConfig(
        provider_type="gcp",
        project_id="test-project",
        zone="us-central1-a",
        # controller_vm left empty - no gcp or manual set
    )
    with pytest.raises(ValueError, match="No controller config specified"):
        create_controller(config)


class TestConfigParsing:
    """Tests for config parsing with controller VM settings."""

    def test_load_config_with_manual_controller(self, tmp_path: Path):
        """Config with controller_vm.manual is parsed correctly."""
        config_content = """\
provider_type: manual

docker_image: gcr.io/project/iris-worker:latest

controller_vm:
  manual:
    host: 10.0.0.100
    image: gcr.io/project/iris-controller:latest
    port: 10000

ssh_user: ubuntu
ssh_private_key: ~/.ssh/id_rsa

manual_hosts:
  - 10.0.0.1
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.controller_vm.manual.host == "10.0.0.100"
        assert config.controller_vm.manual.image == "gcr.io/project/iris-controller:latest"
        assert config.controller_vm.manual.port == 10000
        assert config.ssh_user == "ubuntu"


class TestConfigSerialization:
    """Tests for config serialization via protobuf."""

    def test_config_to_dict_includes_scale_groups(self, tmp_path: Path):
        """config_to_dict() properly serializes scale groups."""
        config_content = """\
provider_type: tpu
project_id: my-project
zone: us-central1-a

docker_image: gcr.io/project/iris-worker:latest

controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    accelerator_type: v5litepod-8
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
    preemptible: true
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        d = config_to_dict(config)

        assert d["provider_type"] == "tpu"
        assert d["project_id"] == "my-project"
        assert d["docker_image"] == "gcr.io/project/iris-worker:latest"
        assert "tpu_v5e_8" in d["scale_groups"]
        sg = d["scale_groups"]["tpu_v5e_8"]
        assert sg["accelerator_type"] == "v5litepod-8"
        assert sg["min_slices"] == 1
        assert sg["max_slices"] == 10
        assert sg["preemptible"] is True

    def test_config_to_dict_round_trips_through_yaml(self, tmp_path: Path):
        """Config serialized to YAML can be loaded back."""
        import yaml

        config_content = """\
provider_type: tpu
project_id: my-project
zone: us-central1-a

docker_image: gcr.io/project/iris-worker:latest
worker_port: 10001

controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    accelerator_type: v5litepod-8
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
    preemptible: true
    priority: 50
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        original_config = load_config(config_path)
        yaml_str = yaml.dump(config_to_dict(original_config), default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        assert loaded_config.provider_type == original_config.provider_type
        assert loaded_config.project_id == original_config.project_id
        assert loaded_config.docker_image == original_config.docker_image
        assert len(loaded_config.scale_groups) == len(original_config.scale_groups)
        assert "tpu_v5e_8" in loaded_config.scale_groups
        assert loaded_config.scale_groups["tpu_v5e_8"].priority == 50


class TestBootstrapScriptConfig:
    """Tests for bootstrap script config injection."""

    def test_bootstrap_script_includes_config_when_provided(self):
        """Bootstrap script writes config file when provided."""
        from iris.cluster.vm.controller import _build_controller_bootstrap_script

        config_yaml = "provider_type: tpu\n"
        script = _build_controller_bootstrap_script(
            docker_image="gcr.io/project/iris:latest",
            port=10000,
            config_yaml=config_yaml,
        )

        assert "/etc/iris/config.yaml" in script
        assert "IRIS_CONFIG_EOF" in script
        assert "--config /etc/iris/config.yaml" in script
        assert "-v /etc/iris/config.yaml:/etc/iris/config.yaml:ro" in script

    def test_bootstrap_script_omits_config_when_empty(self):
        """Bootstrap script skips config setup when not provided."""
        from iris.cluster.vm.controller import _build_controller_bootstrap_script

        script = _build_controller_bootstrap_script(
            docker_image="gcr.io/project/iris:latest",
            port=10000,
            config_yaml="",
        )

        assert "IRIS_CONFIG_EOF" not in script
        assert "--config" not in script
        assert "# No config file provided" in script


class TestHealthCheckIntegration:
    """Tests for SSH-based health checking across controller types."""

    @patch("iris.cluster.vm.controller.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller.DirectSshConnection")
    def test_manual_controller_start_checks_health(
        self,
        mock_conn_cls: MagicMock,
        mock_run_streaming: MagicMock,
        mock_ssh_health: MagicMock,
    ):
        """ManualController.start() calls wait_healthy_via_ssh."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_ssh_health.return_value = True

        config = vm_pb2.IrisClusterConfig(
            provider_type="manual",
            controller_vm=vm_pb2.ControllerVmConfig(
                manual=vm_pb2.ManualControllerConfig(
                    host="10.0.0.100",
                    image="gcr.io/project/iris-controller:latest",
                    port=10000,
                ),
            ),
            ssh_user="ubuntu",
            ssh_private_key="/home/ubuntu/.ssh/id_rsa",
        )
        controller = ManualController(config)
        controller.start()

        mock_ssh_health.assert_called_once()
        call_args = mock_ssh_health.call_args
        assert call_args[0][1] == 10000  # port argument

    @patch("iris.cluster.vm.controller.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller.DirectSshConnection")
    def test_manual_controller_start_fails_on_health_timeout(
        self,
        mock_conn_cls: MagicMock,
        mock_run_streaming: MagicMock,
        mock_ssh_health: MagicMock,
    ):
        """ManualController.start() raises when health check fails."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_ssh_health.return_value = False  # Health check fails

        config = vm_pb2.IrisClusterConfig(
            provider_type="manual",
            controller_vm=vm_pb2.ControllerVmConfig(
                manual=vm_pb2.ManualControllerConfig(
                    host="10.0.0.100",
                    image="gcr.io/project/iris-controller:latest",
                    port=10000,
                ),
            ),
            ssh_user="ubuntu",
            ssh_private_key="/home/ubuntu/.ssh/id_rsa",
        )
        controller = ManualController(config)

        with pytest.raises(RuntimeError, match="failed health check after bootstrap"):
            controller.start()

    @patch("iris.cluster.vm.controller.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller.GceSshConnection")
    def test_gcp_controller_start_checks_health(
        self,
        mock_gce_conn_cls: MagicMock,
        mock_ssh_health: MagicMock,
    ):
        """GcpController.start() calls wait_healthy_via_ssh after VM creation."""
        mock_conn = MagicMock()
        mock_gce_conn_cls.return_value = mock_conn
        mock_ssh_health.return_value = True

        config = vm_pb2.IrisClusterConfig(
            provider_type="gcp",
            project_id="my-project",
            zone="us-central1-a",
            controller_vm=vm_pb2.ControllerVmConfig(
                gcp=vm_pb2.GcpControllerConfig(
                    image="gcr.io/project/iris-controller:latest",
                    port=10000,
                    machine_type="n2-standard-4",
                ),
            ),
        )
        controller = GcpController(config)

        # Mock the _create_vm method to return an address without actually calling gcloud
        with patch.object(controller, "_create_vm", return_value="http://10.0.0.50:10000"):
            with patch.object(controller, "discover", return_value=None):
                with patch.object(controller, "_tag_metadata"):
                    result = controller.start()

        assert result == "http://10.0.0.50:10000"
        mock_ssh_health.assert_called_once()
        call_args = mock_ssh_health.call_args
        assert call_args[0][1] == 10000  # port argument

    @patch("iris.cluster.vm.controller.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller.GceSshConnection")
    def test_gcp_controller_reload_checks_health(
        self,
        mock_gce_conn_cls: MagicMock,
        mock_run_streaming: MagicMock,
        mock_ssh_health: MagicMock,
    ):
        """GcpController.reload() calls wait_healthy_via_ssh after restarting container."""
        mock_conn = MagicMock()
        mock_gce_conn_cls.return_value = mock_conn
        mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_ssh_health.return_value = True

        config = vm_pb2.IrisClusterConfig(
            provider_type="gcp",
            project_id="my-project",
            zone="us-central1-a",
            controller_vm=vm_pb2.ControllerVmConfig(
                gcp=vm_pb2.GcpControllerConfig(
                    image="gcr.io/project/iris-controller:latest",
                    port=10000,
                    machine_type="n2-standard-4",
                ),
            ),
        )
        controller = GcpController(config)

        # Mock the methods that query GCP
        with patch.object(controller, "_find_controller_vm_name", return_value="iris-controller-test"):
            with patch.object(controller, "_get_vm_address", return_value="http://10.0.0.50:10000"):
                result = controller.reload()

        assert result == "http://10.0.0.50:10000"
        mock_ssh_health.assert_called_once()
