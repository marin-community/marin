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

import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from iris.cli import iris
from iris.cluster.vm.config import config_to_dict, load_config
from iris.cluster.vm.controller import (
    GcpController,
    HealthCheckResult,
    ManualController,
    create_controller,
)
from iris.cluster.vm.managed_vm import (
    BOOTSTRAP_SCRIPT,
    _build_bootstrap_script,
    _build_env_flags,
)
from iris.rpc import config_pb2


@pytest.fixture
def ssh_bootstrap_config() -> config_pb2.IrisClusterConfig:
    """Config for SSH bootstrap mode."""
    return config_pb2.IrisClusterConfig(
        provider_type="manual",
        controller_vm=config_pb2.ControllerVmConfig(
            image="gcr.io/project/iris-controller:latest",
            manual=config_pb2.ManualControllerConfig(
                host="10.0.0.100",
                port=10000,
            ),
        ),
        ssh=config_pb2.SshConfig(
            user="ubuntu",
            key_file="/home/ubuntu/.ssh/id_rsa",
            connect_timeout=30,
        ),
    )


@pytest.fixture
def gcp_config() -> config_pb2.IrisClusterConfig:
    """Config for GCP controller VM."""
    return config_pb2.IrisClusterConfig(
        provider_type="gcp",
        project_id="my-project",
        zone="us-central1-a",
        controller_vm=config_pb2.ControllerVmConfig(
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
    config.controller_vm.image = ""

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
    ssh_bootstrap_config: config_pb2.IrisClusterConfig,
):
    """start() SSHs into host and runs bootstrap script."""
    mock_conn = MagicMock()
    mock_conn_cls.return_value = mock_conn
    mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_health.return_value = HealthCheckResult(healthy=True)

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
    call_args = mock_run_streaming.call_args
    command = call_args[0][1]
    assert "docker" in command
    assert "iris-controller" in command


def test_create_controller_raises_on_missing_config(gcp_config: config_pb2.IrisClusterConfig):
    """create_controller raises ValueError when no oneof is set."""
    config = config_pb2.IrisClusterConfig()
    config.CopyFrom(gcp_config)
    config.controller_vm.ClearField("gcp")

    with pytest.raises(ValueError, match="No controller config specified"):
        create_controller(config)


class TestConfigParsing:
    """Tests for config parsing with controller VM settings."""

    def test_load_config_with_manual_controller(self, tmp_path: Path):
        """Config with controller_vm.manual can be loaded and used to create a controller."""
        config_content = """\
provider_type: manual

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001

controller_vm:
  image: gcr.io/project/iris-controller:latest
  manual:
    host: 10.0.0.100
    port: 10000

ssh:
  user: ubuntu
  key_file: ~/.ssh/id_rsa

scale_groups:
  manual_hosts:
    provider:
      manual:
        hosts: [10.0.0.1]
    accelerator_type: cpu
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        # Test behavior: we can create a controller from this config without errors
        create_controller(config)


class TestConfigSerialization:
    """Tests for config serialization via protobuf."""

    def test_config_to_dict_includes_scale_groups(self, tmp_path: Path):
        """config_to_dict() properly serializes scale groups."""
        config_content = """\
provider_type: tpu
project_id: my-project
zone: us-central1-a

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
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
        assert d["bootstrap"]["docker_image"] == "gcr.io/project/iris-worker:latest"
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
provider_type: tpu
project_id: my-project
zone: us-central1-a

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
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
        assert loaded_config.bootstrap.docker_image == original_config.bootstrap.docker_image
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


class TestControllerLifecycle:
    """Tests for controller lifecycle behavior (start, stop, reload)."""

    @pytest.mark.parametrize(
        "controller_type,config_fixture,expected_url",
        [
            ("manual", "ssh_bootstrap_config", "http://10.0.0.100:10000"),
            ("gcp", "gcp_config", "http://10.0.0.50:10000"),
        ],
    )
    @patch("iris.cluster.vm.controller.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller.DirectSshConnection")
    @patch("iris.cluster.vm.controller.GceSshConnection")
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
    @patch("iris.cluster.vm.controller.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller.DirectSshConnection")
    @patch("iris.cluster.vm.controller.GceSshConnection")
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

    @patch("iris.cluster.vm.controller.wait_healthy_via_ssh")
    @patch("iris.cluster.vm.controller.run_streaming_with_retry")
    @patch("iris.cluster.vm.controller.GceSshConnection")
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
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def gcp_config_file(tmp_path: Path) -> Path:
    """Create a GCP controller config file."""
    config_path = tmp_path / "gcp_config.yaml"
    config_content = """
provider_type: gcp
project_id: test-project
region: us-central1
zone: us-central1-a

bootstrap:
  docker_image: test-image:latest
  worker_port: 10001

controller_vm:
  image: gcr.io/test-project/iris-controller:latest
  gcp:
    machine_type: n2-standard-4
    boot_disk_size_gb: 50
    port: 10000

scale_groups:
  test_group:
    accelerator_type: tpu
    accelerator_variant: v5litepod-4
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 0
    max_slices: 10
    zones: [us-central1-a]
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def manual_config_file(tmp_path: Path) -> Path:
    """Create a manual controller config file with SSH bootstrap."""
    config_path = tmp_path / "manual_config.yaml"
    config_content = """
provider_type: manual

bootstrap:
  docker_image: gcr.io/test-project/iris-worker:latest
  worker_port: 10001

controller_vm:
  image: gcr.io/test-project/iris-controller:latest
  manual:
    host: 10.0.0.100
    port: 10000

ssh:
  user: ubuntu
  key_file: ~/.ssh/id_rsa
  connect_timeout: 30

scale_groups:
  manual_hosts:
    provider:
      manual:
        hosts: [10.0.0.1]
    accelerator_type: cpu
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
        controller = create_controller(config)
        assert isinstance(controller, expected_type)


class TestCliControllerCommands:
    """Tests for CLI controller start/stop/reload commands."""

    @pytest.mark.parametrize(
        "command,mock_target,error_msg,expected_output",
        [
            (
                "stop",
                "iris.cluster.vm.controller.GcpController.stop",
                "VM not found",
                "Failed to stop controller: VM not found",
            ),
        ],
    )
    def test_cli_controller_failure_shows_error(
        self,
        cli_runner: CliRunner,
        gcp_config_file: Path,
        command: str,
        mock_target: str,
        error_msg: str,
        expected_output: str,
    ):
        """CLI shows appropriate error message when controller operation fails."""
        with patch(mock_target) as mock_op:
            mock_op.side_effect = RuntimeError(error_msg)

            result = cli_runner.invoke(
                iris,
                ["cluster", "--config", str(gcp_config_file), "controller", command],
            )

        assert result.exit_code == 1
        assert expected_output in result.output


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

    def test_template_has_only_expected_placeholders(self):
        """Template contains only expected placeholders and properly escapes Docker format strings."""
        expected_placeholders = {"cache_dir", "docker_image", "worker_port", "env_flags"}

        # Extract single-brace placeholders (not preceded/followed by more braces)
        pattern = r"(?<!\{)\{([^{}\s]+)\}(?!\})"
        found_placeholders = set(re.findall(pattern, BOOTSTRAP_SCRIPT))
        found_placeholders = {p for p in found_placeholders if not p.startswith(".")}

        assert (
            found_placeholders == expected_placeholders
        ), f"Unexpected placeholders: {found_placeholders - expected_placeholders}"

        # Verify Docker format strings use quadruple braces
        assert "{{{{.Status}}}}" in BOOTSTRAP_SCRIPT
        assert "{{{{.State}}}}" in BOOTSTRAP_SCRIPT

    def test_template_comments_have_no_unescaped_braces(self):
        """Comments should not contain unescaped single braces like {N}."""
        lines = BOOTSTRAP_SCRIPT.split("\n")
        comment_lines = [(i + 1, line) for i, line in enumerate(lines) if "#" in line]
        unescaped_pattern = r"(?<!\{)\{[^{}]+\}(?!\})"

        errors = []
        for line_num, line in comment_lines:
            comment_part = line[line.index("#") :]
            matches = re.findall(unescaped_pattern, comment_part)
            if matches:
                errors.append(f"Line {line_num}: {line.strip()} - Found: {matches}")

        assert not errors, "Found unescaped braces in comments:\n" + "\n".join(errors)

    def test_build_bootstrap_script_no_key_error(self, minimal_bootstrap_config: config_pb2.BootstrapConfig):
        """Template formatting should not raise KeyError."""
        try:
            script = _build_bootstrap_script(minimal_bootstrap_config, vm_address="10.0.0.1")
            assert script
        except KeyError as e:
            pytest.fail(f"Template has unescaped braces: {{{e.args[0]}}}")

    def test_build_bootstrap_script_replaces_all_placeholders(
        self, minimal_bootstrap_config: config_pb2.BootstrapConfig
    ):
        """Generated script should not contain raw placeholders."""
        script = _build_bootstrap_script(minimal_bootstrap_config, vm_address="10.0.0.1")

        for placeholder in ["{cache_dir}", "{docker_image}", "{worker_port}", "{env_flags}"]:
            assert placeholder not in script

    def test_build_bootstrap_script_preserves_docker_format_strings(
        self, minimal_bootstrap_config: config_pb2.BootstrapConfig
    ):
        """Docker format strings become double braces after formatting."""
        script = _build_bootstrap_script(minimal_bootstrap_config, vm_address="10.0.0.1")

        assert "{{.Status}}" in script
        assert "{{.State}}" in script

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

    def test_build_env_flags_includes_controller_address(self, minimal_bootstrap_config: config_pb2.BootstrapConfig):
        """Env flags include IRIS_CONTROLLER_ADDRESS using shell variable."""
        flags = _build_env_flags(minimal_bootstrap_config, vm_address="10.0.0.1")
        assert '-e IRIS_CONTROLLER_ADDRESS="$CONTROLLER_ADDRESS"' in flags

    def test_build_env_flags_includes_tpu_passthroughs(self, minimal_bootstrap_config: config_pb2.BootstrapConfig):
        """Env flags include all TPU passthrough variables."""
        flags = _build_env_flags(minimal_bootstrap_config, vm_address="10.0.0.1")

        for var in ["TPU_NAME", "TPU_TYPE", "TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES", "TPU_CHIPS_PER_HOST_BOUNDS"]:
            assert f'-e {var}="${{{var}:-}}"' in flags

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
