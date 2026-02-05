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

"""Tests for controller bootstrap and config serialization."""

from __future__ import annotations

from pathlib import Path
import pytest

from iris.cluster.platform.bootstrap import (
    _build_env_flags,
    build_controller_bootstrap_script,
    build_worker_bootstrap_script,
)
from iris.cluster.platform.cluster_manager import ClusterManager
from iris.rpc import vm_pb2
from iris.config import config_to_dict, load_config
from iris.rpc import config_pb2
from iris.time_utils import Timestamp


class _FakePlatform:
    def __init__(self) -> None:
        self.started = 0

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[vm_pb2.VmInfo]:
        return []

    def start_vms(self, spec, *, zone: str | None = None) -> list[vm_pb2.VmInfo]:
        self.started += 1
        return [
            vm_pb2.VmInfo(
                vm_id="controller-1",
                address="http://10.0.0.10:10000",
                zone=zone,
                labels=dict(spec.labels),
                state=vm_pb2.VM_STATE_READY,
                created_at=Timestamp.now().to_proto(),
            )
        ]

    def stop_vms(self, ids, *, zone: str | None = None) -> None:
        return None


@pytest.fixture
def manual_controller_config() -> config_pb2.IrisClusterConfig:
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
    )


def test_controller_vm_start_requires_image(manual_controller_config: config_pb2.IrisClusterConfig):
    config = config_pb2.IrisClusterConfig()
    config.CopyFrom(manual_controller_config)
    config.controller.image = ""

    controller = ClusterManager(config, platform=_FakePlatform())
    with pytest.raises(RuntimeError, match=r"controller.image is required"):
        controller.start()


class TestConfigParsing:
    """Tests for config parsing with controller VM settings."""

    def test_load_config_with_manual_controller(self, tmp_path: Path):
        """Config with controller.manual can be loaded."""
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
      gpu_count: 0
      tpu_count: 0
    manual:
      hosts: [10.0.0.1]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.controller.manual.host == "10.0.0.100"
        assert config.controller.manual.port == 10000


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
      gpu_count: 0
      tpu_count: 8
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
      gpu_count: 0
      tpu_count: 8
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


class TestControllerBootstrapScript:
    """Tests for controller bootstrap script config injection."""

    def test_bootstrap_script_includes_config_when_provided(self):
        config_yaml = "platform:\n  gcp:\n    project_id: my-project\n"
        script = build_controller_bootstrap_script(
            docker_image="gcr.io/project/iris:latest",
            port=10000,
            config_yaml=config_yaml,
        )

        assert "/etc/iris/config.yaml" in script
        assert "IRIS_CONFIG_EOF" in script
        assert "--config /etc/iris/config.yaml" in script
        assert "-v /etc/iris/config.yaml:/etc/iris/config.yaml:ro" in script

    def test_bootstrap_script_omits_config_when_empty(self):
        script = build_controller_bootstrap_script(
            docker_image="gcr.io/project/iris:latest",
            port=10000,
            config_yaml="",
        )

        assert "IRIS_CONFIG_EOF" not in script
        assert "--config" not in script
        assert "# No config file provided" in script


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


class TestWorkerBootstrapScript:
    """Tests for worker bootstrap script template and generation functions."""

    def test_build_bootstrap_script_no_key_error(self, minimal_bootstrap_config: config_pb2.BootstrapConfig):
        """Template formatting should not raise KeyError."""
        try:
            script = build_worker_bootstrap_script(minimal_bootstrap_config, vm_address="10.0.0.1")
            assert script
        except KeyError as e:
            pytest.fail(f"Template has unescaped braces: {{{e.args[0]}}}")

    def test_build_bootstrap_script_prepends_discovery_preamble(
        self, minimal_bootstrap_config: config_pb2.BootstrapConfig
    ):
        """Discovery preamble is prepended to script."""
        preamble = "export CONTROLLER_ADDRESS=http://10.0.0.1:10000\n"
        script = build_worker_bootstrap_script(
            minimal_bootstrap_config,
            vm_address="10.0.0.1",
            discovery_preamble=preamble,
        )

        assert script.startswith(preamble)

    def test_build_env_flags_escapes_special_chars(self, config_with_special_chars: config_pb2.BootstrapConfig):
        flags = _build_env_flags(config_with_special_chars, vm_address="10.0.0.1")
        assert "Hello {world}!" in flags
        assert '{"key": "value"}' in flags
