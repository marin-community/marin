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

"""Behavioral tests for controller factory and CLI integration.

These tests verify controller creation logic and error handling
without testing implementation details like command strings.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from iris.cli import iris
from iris.cluster.vm.controller import (
    GcpController,
    ManualController,
    create_controller,
)
from iris.cluster.vm.controller import HealthCheckResult
from iris.rpc import config_pb2


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def gcp_config(tmp_path: Path) -> Path:
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
  gcp:
    image: gcr.io/test-project/iris-controller:latest
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
def manual_config(tmp_path: Path) -> Path:
    """Create a manual controller config file with SSH bootstrap."""
    config_path = tmp_path / "manual_config.yaml"
    config_content = """
provider_type: manual

bootstrap:
  docker_image: gcr.io/test-project/iris-worker:latest
  worker_port: 10001

controller_vm:
  manual:
    host: 10.0.0.100
    image: gcr.io/test-project/iris-controller:latest
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


def test_creates_gcp_controller_for_gcp_provider_with_vm_enabled():
    """create_controller returns GcpController for GCP with gcp config."""
    config = config_pb2.IrisClusterConfig(
        provider_type="gcp",
        project_id="test-project",
        zone="us-central1-a",
        controller_vm=config_pb2.ControllerVmConfig(
            gcp=config_pb2.GcpControllerConfig(
                image="gcr.io/test/controller:latest",
            ),
        ),
    )
    controller = create_controller(config)
    assert isinstance(controller, GcpController)


def test_creates_manual_controller_for_manual_provider():
    """create_controller returns ManualController for manual provider with host."""
    config = config_pb2.IrisClusterConfig(
        provider_type="manual",
        controller_vm=config_pb2.ControllerVmConfig(
            manual=config_pb2.ManualControllerConfig(
                host="10.0.0.100",
                image="gcr.io/test/controller:latest",
            ),
        ),
    )
    controller = create_controller(config)
    assert isinstance(controller, ManualController)


@patch("iris.cluster.vm.controller.check_health")
@patch("iris.cluster.vm.controller.run_streaming_with_retry")
@patch("iris.cluster.vm.controller.DirectSshConnection")
def test_cli_controller_start_with_ssh(
    mock_conn_cls: MagicMock,
    mock_run_streaming: MagicMock,
    mock_health: MagicMock,
    cli_runner: CliRunner,
    manual_config: Path,
):
    """CLI controller start with SSH bootstrap runs bootstrap script."""
    mock_conn = MagicMock()
    mock_conn_cls.return_value = mock_conn
    mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_health.return_value = HealthCheckResult(healthy=True)

    result = cli_runner.invoke(
        iris,
        ["cluster", "--config", str(manual_config), "controller", "start", "--skip-build"],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Controller started successfully at http://10.0.0.100:10000" in result.output


def test_cli_start_failure_shows_error(
    cli_runner: CliRunner,
    gcp_config: Path,
):
    """CLI shows error message when controller start fails."""

    def mock_run_fail(cmd, **_kwargs):
        result = MagicMock(spec=subprocess.CompletedProcess)
        if "create" in cmd:
            result.returncode = 1
            result.stderr = "Quota exceeded"
        else:
            result.returncode = 0
            result.stdout = ""
        result.stdout = result.stdout if hasattr(result, "stdout") else ""
        result.stderr = result.stderr if hasattr(result, "stderr") else ""
        return result

    with patch("subprocess.run", side_effect=mock_run_fail):
        with patch("iris.cluster.vm.controller._check_health_rpc", return_value=False):
            with patch("iris.cluster.vm.controller.time.sleep"):
                result = cli_runner.invoke(
                    iris,
                    ["cluster", "--config", str(gcp_config), "controller", "start", "--skip-build"],
                )

    assert result.exit_code == 1
    assert "Failed to start controller" in result.output


def test_cli_stop_failure_shows_error(
    cli_runner: CliRunner,
    gcp_config: Path,
):
    """CLI shows error message when controller stop fails."""
    with patch("iris.cluster.vm.controller.GcpController.stop") as mock_stop:
        mock_stop.side_effect = RuntimeError("VM not found")

        result = cli_runner.invoke(
            iris,
            ["cluster", "--config", str(gcp_config), "controller", "stop"],
        )

    assert result.exit_code == 1
    assert "Failed to stop controller: VM not found" in result.output


@patch("iris.cluster.vm.controller.check_health")
@patch("iris.cluster.vm.controller.run_streaming_with_retry")
@patch("iris.cluster.vm.controller.DirectSshConnection")
def test_manual_start_timeout_shows_error(
    mock_conn_cls: MagicMock,
    mock_run_streaming: MagicMock,
    mock_health: MagicMock,
    cli_runner: CliRunner,
    manual_config: Path,
):
    """CLI shows error when manual controller health check times out."""
    mock_conn = MagicMock()
    mock_conn_cls.return_value = mock_conn
    mock_run_streaming.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mock_health.return_value = False  # Health check always fails

    # Simulate time progressing past timeout
    time_calls = [0]

    def mock_monotonic():
        result = time_calls[0]
        time_calls[0] += 200  # Jump past timeout each call
        return result

    with patch("time.sleep"):
        with patch("time.monotonic", side_effect=mock_monotonic):
            result = cli_runner.invoke(
                iris,
                ["cluster", "--config", str(manual_config), "controller", "start", "--skip-build"],
            )

    assert result.exit_code == 1
    assert "Failed to start controller" in result.output
