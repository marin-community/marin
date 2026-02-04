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
from iris.cluster.vm.controller_vm import (
    GcpController,
    HealthCheckResult,
    ManualController,
    check_health,
    create_controller_vm,
)
from iris.cluster.vm.managed_vm import (
    _build_bootstrap_script,
    _build_env_flags,
)
from iris.rpc import config_pb2
from iris.time_utils import Duration


@pytest.fixture
def ssh_bootstrap_config() -> config_pb2.IrisClusterConfig:
    """Config for SSH bootstrap mode."""
    ssh_config = config_pb2.SshConfig(
        user="ubuntu",
        key_file="/home/ubuntu/.ssh/id_rsa",
    )
    ssh_config.connect_timeout.CopyFrom(Duration.from_seconds(30).to_proto())

    return config_pb2.IrisClusterConfig(
        platform=config_pb2.PlatformConfig(
            manual=config_pb2.ManualPlatformConfig(),
        ),
        controller=config_pb2.ControllerVmConfig(
            image="gcr.io/project/iris-controller:latest",
            manual=config_pb2.ManualControllerConfig(
                host="10.0.0.100",
                port=10000,
            ),
        ),
        defaults=config_pb2.DefaultsConfig(
            ssh=ssh_config,
        ),
    )


@pytest.fixture
def gcp_config() -> config_pb2.IrisClusterConfig:
    """Config for GCP controller VM."""
    return config_pb2.IrisClusterConfig(
        platform=config_pb2.PlatformConfig(
            gcp=config_pb2.GcpPlatformConfig(
                project_id="my-project",
                zone="us-central1-a",
            ),
        ),
        controller=config_pb2.ControllerVmConfig(
            image="gcr.io/project/iris-controller:latest",
            gcp=config_pb2.GcpControllerConfig(
                port=10000,
            ),
        ),
    )


def test_manual_controller_start_requires_image(ssh_bootstrap_config: config_pb2.IrisClusterConfig):
    """start() requires image to be configured."""
    config = config_pb2.IrisClusterConfig()
    config.CopyFrom(ssh_bootstrap_config)
    config.controller.image = ""

    controller = ManualController(config)
    with pytest.raises(RuntimeError, match="image required"):
        controller.start()


@patch("iris.cluster.vm.controller_vm.check_health")
@patch("iris.cluster.vm.controller_vm.run_streaming_with_retry")
@patch("iris.cluster.vm.controller_vm.DirectSshConnection")
def test_manual_controller_start_runs_bootstrap(
    mock_conn_cls: MagicMock,
    mock_run_streaming: MagicMock,
    mock_health: MagicMock,
    ssh_bootstrap_config: config_pb2.IrisClusterConfig,
):
    """start() returns correct URL and successfully bootstraps when healthy."""
    mock_conn = MagicMock()
    mock_conn_cls.return_value = mock_conn
    mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_health.return_value = HealthCheckResult(healthy=True)

    controller = ManualController(ssh_bootstrap_config)
    result = controller.start()

    assert result == "http://10.0.0.100:10000"


def test_create_controller_raises_on_missing_config(gcp_config: config_pb2.IrisClusterConfig):
    """create_controller raises ValueError when no oneof is set."""
    config = config_pb2.IrisClusterConfig()
    config.CopyFrom(gcp_config)
    config.controller.ClearField("gcp")

    with pytest.raises(ValueError, match="No controller config specified"):
        create_controller_vm(config)


class TestConfigParsing:
    """Tests for config parsing with controller VM settings."""

    def test_load_config_with_manual_controller(self, tmp_path: Path):
        """Config with controller.manual can be loaded and used to create a controller."""
        config_content = """\
platform:
  manual: {}

defaults:
  bootstrap:
    docker_image: gcr.io/project/iris-worker:latest
    worker_port: 10001
  ssh:
    user: ubuntu
    key_file: ~/.ssh/id_rsa

controller:
  image: gcr.io/project/iris-controller:latest
  manual:
    host: 10.0.0.100
    port: 10000

scale_groups:
  manual_hosts:
    vm_type: manual_vm
    accelerator_type: cpu
    slice_size: 1
    resources:
      cpu: 1
      ram: 1GB
      disk: 0
      gpu: 0
      tpu: 0
    manual:
      hosts: [10.0.0.1]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        # Test behavior: we can create a controller from this config without errors
        create_controller_vm(config)


class TestConfigSerialization:
    """Tests for config serialization via protobuf."""

    def test_config_to_dict_includes_scale_groups(self, tmp_path: Path):
        """config_to_dict() properly serializes scale groups."""
        config_content = """\
platform:
  gcp:
    project_id: my-project
    zone: us-central1-a

defaults:
  bootstrap:
    docker_image: gcr.io/project/iris-worker:latest
    worker_port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
    preemptible: true
    slice_size: 8
    resources:
      cpu: 1
      ram: 1GB
      disk: 0
      gpu: 0
      tpu: 8
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        d = config_to_dict(config)

        assert d["platform"]["gcp"]["project_id"] == "my-project"
        assert d["defaults"]["bootstrap"]["docker_image"] == "gcr.io/project/iris-worker:latest"
        assert "tpu_v5e_8" in d["scale_groups"]
        sg = d["scale_groups"]["tpu_v5e_8"]
        assert sg["accelerator_variant"] == "v5litepod-8"
        assert sg["min_slices"] == 1
        assert sg["max_slices"] == 10
        assert sg["preemptible"] is True

    def test_config_to_dict_round_trips_through_yaml(self, tmp_path: Path):
        """Config serialized to YAML can be loaded back."""
        import yaml

        config_content = """\
platform:
  gcp:
    project_id: my-project
    zone: us-central1-a

defaults:
  bootstrap:
    docker_image: gcr.io/project/iris-worker:latest
    worker_port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
    preemptible: true
    priority: 50
    slice_size: 8
    resources:
      cpu: 1
      ram: 1GB
      disk: 0
      gpu: 0
      tpu: 8
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        original_config = load_config(config_path)
        yaml_str = yaml.dump(config_to_dict(original_config), default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        assert loaded_config.platform == original_config.platform
        assert loaded_config.defaults.bootstrap.docker_image == original_config.defaults.bootstrap.docker_image
        assert len(loaded_config.scale_groups) == len(original_config.scale_groups)
        assert "tpu_v5e_8" in loaded_config.scale_groups
        assert loaded_config.scale_groups["tpu_v5e_8"].priority == 50


class TestBootstrapScriptConfig:
    """Tests for bootstrap script config injection."""

    def test_bootstrap_script_includes_config_when_provided(self):
        """Bootstrap script writes config file when provided."""
        from iris.cluster.vm.controller_vm import _build_controller_bootstrap_script

        config_yaml = "platform:\\n  gcp:\\n    project_id: my-project\\n"
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
        from iris.cluster.vm.controller_vm import _build_controller_bootstrap_script

        script = _build_controller_bootstrap_script(
            docker_image="gcr.io/project/iris:latest",
            port=10000,
            config_yaml="",
        )

        assert "IRIS_CONFIG_EOF" not in script
        assert "--config" not in script
        assert "# No config file provided" in script


class TestControllerLifecycle:
    """Tests for controller lifecycle behavior (start, stop, reload)."""

    @pytest.mark.parametrize(
        "controller_type,config_fixture,expected_url",
        [
            ("manual", "ssh_bootstrap_config", "http://10.0.0.100:10000"),
            ("gcp", "gcp_config", "http://10.0.0.50:10000"),
        ],
    )
    @patch("iris.cluster.vm.controller_vm.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller_vm.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller_vm.DirectSshConnection")
    @patch("iris.cluster.vm.controller_vm.GceSshConnection")
    def test_controller_start_succeeds_when_healthy(
        self,
        mock_gce_conn: MagicMock,
        mock_direct_conn: MagicMock,
        mock_run_streaming: MagicMock,
        mock_wait_healthy: MagicMock,
        controller_type: str,
        config_fixture: str,
        expected_url: str,
        request,
    ):
        """Controller.start() returns URL when controller becomes healthy."""
        mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_wait_healthy.return_value = True

        config = request.getfixturevalue(config_fixture)

        if controller_type == "manual":
            controller = ManualController(config)
            result = controller.start()
        else:  # gcp
            controller = GcpController(config)
            with patch.object(controller, "_create_vm", return_value=expected_url):
                with patch.object(controller, "discover", return_value=None):
                    with patch.object(controller, "_tag_metadata"):
                        result = controller.start()

        assert result == expected_url

    @pytest.mark.parametrize(
        "controller_type,config_fixture",
        [
            ("manual", "ssh_bootstrap_config"),
            ("gcp", "gcp_config"),
        ],
    )
    @patch("iris.cluster.vm.controller_vm.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller_vm.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller_vm.DirectSshConnection")
    @patch("iris.cluster.vm.controller_vm.GceSshConnection")
    def test_controller_start_fails_when_unhealthy(
        self,
        mock_gce_conn: MagicMock,
        mock_direct_conn: MagicMock,
        mock_run_streaming: MagicMock,
        mock_wait_healthy: MagicMock,
        controller_type: str,
        config_fixture: str,
        request,
    ):
        """Controller.start() raises RuntimeError when health check fails."""
        mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_wait_healthy.return_value = False

        config = request.getfixturevalue(config_fixture)

        if controller_type == "manual":
            controller = ManualController(config)
            with pytest.raises(RuntimeError, match="failed health check after bootstrap"):
                controller.start()
        else:  # gcp
            controller = GcpController(config)
            with patch.object(controller, "_create_vm", return_value="http://10.0.0.50:10000"):
                with patch.object(controller, "discover", return_value=None):
                    with patch.object(controller, "_tag_metadata"):
                        with pytest.raises(RuntimeError, match="failed health check after bootstrap"):
                            controller.start()

    @patch("iris.cluster.vm.controller_vm.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller_vm.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller_vm.GceSshConnection")
    def test_gcp_controller_reload_returns_url_when_healthy(
        self,
        mock_gce_conn: MagicMock,
        mock_run_streaming: MagicMock,
        mock_wait_healthy: MagicMock,
        gcp_config: config_pb2.IrisClusterConfig,
    ):
        """GcpController.reload() returns URL after successfully restarting container."""
        mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_wait_healthy.return_value = True

        controller = GcpController(gcp_config)

        with patch.object(controller, "_find_controller_vm_name", return_value="iris-controller-test"):
            with patch.object(controller, "_get_vm_address", return_value="http://10.0.0.50:10000"):
                result = controller.reload()

        assert result == "http://10.0.0.50:10000"


# ============================================================================
# CLI Integration Tests (from test_controller_boot.py)
# ============================================================================


@pytest.fixture
def manual_config_file(tmp_path: Path) -> Path:
    """Create a manual controller config file with SSH bootstrap."""
    config_path = tmp_path / "manual_config.yaml"
    config_content = """
platform:
  manual: {}

defaults:
  bootstrap:
    docker_image: gcr.io/test-project/iris-worker:latest
    worker_port: 10001
  ssh:
    user: ubuntu
    key_file: ~/.ssh/id_rsa
    connect_timeout: { milliseconds: 30000 }

controller:
  image: gcr.io/test-project/iris-controller:latest
  manual:
    host: 10.0.0.100
    port: 10000

scale_groups:
  manual_hosts:
    vm_type: manual_vm
    accelerator_type: cpu
    manual:
      hosts: [10.0.0.1]
"""
    config_path.write_text(config_content)
    return config_path


class TestControllerFactory:
    """Tests for controller factory and type selection."""

    @pytest.mark.parametrize(
        "config_fixture,expected_type",
        [
            ("gcp_config", GcpController),
            ("ssh_bootstrap_config", ManualController),
        ],
    )
    def test_creates_correct_controller_type(self, config_fixture: str, expected_type: type, request):
        """create_controller returns correct controller type based on config."""
        config = request.getfixturevalue(config_fixture)
        controller = create_controller_vm(config)
        assert isinstance(controller, expected_type)


# ============================================================================
# Worker Bootstrap Script Tests
# ============================================================================


@pytest.fixture
def minimal_bootstrap_config() -> config_pb2.BootstrapConfig:
    """Minimal valid bootstrap config."""
    return config_pb2.BootstrapConfig(
        docker_image="gcr.io/test/iris-worker:latest",
        worker_port=10001,
        cache_dir="/var/cache/iris",
    )


@pytest.fixture
def config_with_special_chars() -> config_pb2.BootstrapConfig:
    """Config with values containing braces and special characters."""
    return config_pb2.BootstrapConfig(
        docker_image="gcr.io/test/iris:v1.0-{tag}",
        cache_dir="/cache/{project}/iris",
        worker_port=10001,
        env_vars={
            "MESSAGE": "Hello {world}!",
            "JSON": '{"key": "value"}',
        },
    )


class TestBootstrapScript:
    """Tests for worker bootstrap script template and generation functions."""

    def test_build_bootstrap_script_no_key_error(self, minimal_bootstrap_config: config_pb2.BootstrapConfig):
        """Template formatting should not raise KeyError."""
        try:
            script = _build_bootstrap_script(minimal_bootstrap_config, vm_address="10.0.0.1")
            assert script
        except KeyError as e:
            pytest.fail(f"Template has unescaped braces: {{{e.args[0]}}}")

    def test_build_bootstrap_script_prepends_discovery_preamble(
        self, minimal_bootstrap_config: config_pb2.BootstrapConfig
    ):
        """Discovery preamble is prepended to script."""
        preamble = "export CONTROLLER_ADDRESS=http://10.0.0.1:10000\n"
        script = _build_bootstrap_script(
            minimal_bootstrap_config,
            vm_address="10.0.0.1",
            discovery_preamble=preamble,
        )

        assert script.startswith(preamble)

    # Environment flags tests

    def test_build_env_flags_quotes_values_with_spaces(self):
        """Env var values with spaces are properly quoted."""
        config = config_pb2.BootstrapConfig(
            docker_image="gcr.io/test/iris:latest",
            worker_port=10001,
            env_vars={"MSG": "hello world"},
        )
        flags = _build_env_flags(config, vm_address="10.0.0.1")
        assert "MSG='hello world'" in flags or 'MSG="hello world"' in flags

    # Edge cases

    def test_braces_in_config_values_preserved(self, config_with_special_chars: config_pb2.BootstrapConfig):
        """Config values with braces are preserved in output."""
        script = _build_bootstrap_script(config_with_special_chars, vm_address="10.0.0.1")

        assert "v1.0-{tag}" in script
        assert "/cache/{project}/iris" in script
        assert "Hello {world}!" in script
        assert '{"key": "value"}' in script

    @pytest.mark.parametrize(
        "value", ["value{N}more", "{start}middle{end}", 'json:{"key":"val"}', "{{escaped}}", "{single}"]
    )
    def test_problematic_patterns_dont_break_formatting(self, value: str):
        """Various brace patterns in config values don't cause errors."""
        config = config_pb2.BootstrapConfig(
            docker_image=f"gcr.io/test/iris:{value}",
            worker_port=10001,
            cache_dir="/var/cache/iris",
        )

        try:
            script = _build_bootstrap_script(config, vm_address="10.0.0.1")
            assert script
            assert value in script
        except KeyError as e:
            pytest.fail(f"Value '{value}' caused KeyError: {e}")

    # Regression tests

    # Integration tests

    def test_multiple_configs_generate_without_errors(self):
        """Various config combinations generate scripts without errors."""
        configs = [
            config_pb2.BootstrapConfig(docker_image="gcr.io/test/iris:latest", worker_port=10001),
            config_pb2.BootstrapConfig(
                docker_image="gcr.io/test/iris:latest",
                worker_port=10001,
                env_vars={"KEY": "value"},
            ),
            config_pb2.BootstrapConfig(
                docker_image="gcr.io/test/iris:v1.0-{tag}",
                cache_dir="/cache/{project}",
                worker_port=10001,
                env_vars={"MSG": "Hello {world}!", "JSON": '{"key": "value"}'},
            ),
        ]

        for config in configs:
            for vm_addr in ["10.0.0.1", ""]:
                for preamble in ["", "export CONTROLLER_ADDRESS=http://10.0.0.1:10000\n"]:
                    script = _build_bootstrap_script(config, vm_address=vm_addr, discovery_preamble=preamble)
                    assert script
                    # No raw placeholders should remain
                    for placeholder in ["{cache_dir}", "{docker_image}", "{worker_port}", "{env_flags}"]:
                        assert placeholder not in script


def test_check_health_passes_duration_to_conn_run():
    """check_health passes Duration objects (not bare ints) to conn.run timeout."""
    mock_conn = MagicMock()
    mock_conn.run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="healthy", stderr="")

    result = check_health(mock_conn, port=8080, container_name="test-container")

    assert result.healthy
    for call in mock_conn.run.call_args_list:
        timeout_arg = call.kwargs.get("timeout")
        assert isinstance(
            timeout_arg, Duration
        ), f"conn.run called with timeout={timeout_arg!r} (type {type(timeout_arg).__name__}), expected Duration"
