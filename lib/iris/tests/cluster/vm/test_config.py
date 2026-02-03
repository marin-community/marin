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

"""Tests for config loading, serialization, and deserialization.

These tests focus on stable behavior of config round-tripping through
YAML, ensuring that vm_type and platform configuration are preserved correctly.
"""

from pathlib import Path

import pytest
import yaml
from google.protobuf.json_format import ParseDict

from iris.cluster.vm.config import (
    config_to_dict,
    create_autoscaler,
    get_ssh_config,
    load_config,
)
from iris.cluster.vm.platform import create_platform
from iris.rpc import config_pb2


class TestConfigRoundTrip:
    """Tests for config serialization/deserialization round-trips."""

    def test_tpu_provider_survives_round_trip(self, tmp_path: Path):
        """TPU vm_type survives proto→dict→yaml→dict→proto round-trip."""
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
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        # Load config from YAML
        original_config = load_config(config_path)

        # Verify vm_type before round-trip
        assert original_config.scale_groups["tpu_v5e_8"].vm_type == config_pb2.VM_TYPE_TPU_VM

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify vm_type is still TPU after round-trip
        assert loaded_config.scale_groups["tpu_v5e_8"].vm_type == config_pb2.VM_TYPE_TPU_VM

    def test_manual_provider_survives_round_trip(self, tmp_path: Path):
        """Manual vm_type survives proto→dict→yaml→dict→proto round-trip."""
        config_content = """\
platform:
  manual: {}

defaults:
  bootstrap:
    docker_image: gcr.io/project/iris-worker:latest
    worker_port: 10001
    controller_address: "http://10.0.0.1:10000"
  ssh:
    user: ubuntu
    key_file: ~/.ssh/id_rsa

scale_groups:
  manual_hosts:
    vm_type: manual_vm
    accelerator_type: cpu
    manual:
      hosts: [10.0.0.1, 10.0.0.2]
      ssh_user: ubuntu
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        # Load config from YAML
        original_config = load_config(config_path)

        # Verify vm_type is manual before round-trip
        assert original_config.scale_groups["manual_hosts"].vm_type == config_pb2.VM_TYPE_MANUAL_VM
        assert list(original_config.scale_groups["manual_hosts"].manual.hosts) == ["10.0.0.1", "10.0.0.2"]

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify vm_type is still manual after round-trip
        assert loaded_config.scale_groups["manual_hosts"].vm_type == config_pb2.VM_TYPE_MANUAL_VM
        assert list(loaded_config.scale_groups["manual_hosts"].manual.hosts) == ["10.0.0.1", "10.0.0.2"]

    def test_multiple_scale_groups_preserve_vm_types(self, tmp_path: Path):
        """Config with multiple TPU scale groups preserves vm_type values."""
        config_content = """\
platform:
  gcp:
    project_id: my-project
    zone: us-central1-a

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_group_a:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
  tpu_group_b:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 0
    max_slices: 4
    zones: [us-central1-a]
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        original_config = load_config(config_path)

        # Round-trip
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        assert loaded_config.scale_groups["tpu_group_a"].vm_type == config_pb2.VM_TYPE_TPU_VM
        assert loaded_config.scale_groups["tpu_group_b"].vm_type == config_pb2.VM_TYPE_TPU_VM

    def test_example_eu_west4_config_round_trips(self):
        """Real example config from examples/eu-west4.yaml round-trips correctly."""
        iris_root = Path(__file__).parent.parent.parent.parent
        config_path = iris_root / "examples" / "eu-west4.yaml"
        if not config_path.exists():
            pytest.skip("Example config not found")

        original_config = load_config(config_path)

        # Verify it has TPU vm_type before round-trip
        assert "tpu_v5e_16" in original_config.scale_groups
        assert original_config.scale_groups["tpu_v5e_16"].vm_type == config_pb2.VM_TYPE_TPU_VM

        # Round-trip via dict and ParseDict
        config_dict = config_to_dict(original_config)
        loaded_config = ParseDict(config_dict, config_pb2.IrisClusterConfig())

        # Verify vm_type is still TPU
        assert loaded_config.scale_groups["tpu_v5e_16"].vm_type == config_pb2.VM_TYPE_TPU_VM

    @pytest.mark.parametrize(
        "accelerator_type,expected_enum",
        [
            ("tpu", config_pb2.ACCELERATOR_TYPE_TPU),
            ("cpu", config_pb2.ACCELERATOR_TYPE_CPU),
            ("gpu", config_pb2.ACCELERATOR_TYPE_GPU),
        ],
    )
    def test_lowercase_accelerator_types_work(self, tmp_path: Path, accelerator_type: str, expected_enum):
        """Config accepts lowercase accelerator types and converts them to enum values."""
        config_content = f"""\
platform:
  manual: {{}}

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

scale_groups:
  test_group:
    vm_type: manual_vm
    accelerator_type: {accelerator_type}
    manual:
      hosts: [10.0.0.1]
    zones: [local]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.scale_groups["test_group"].accelerator_type == expected_enum

    def test_uppercase_accelerator_types_still_work(self, tmp_path: Path):
        """Config still accepts uppercase accelerator types for backwards compatibility."""
        config_content = """\
platform:
  gcp:
    project_id: my-project
    zone: us-central1-a

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_group:
    vm_type: tpu_vm
    accelerator_type: TPU
    accelerator_variant: v5litepod-8
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.scale_groups["tpu_group"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU


class TestCreateAutoscalerFromConfig:
    """Tests for create_autoscaler factory function."""

    def test_creates_autoscaler_with_tpu_provider(self, tmp_path: Path):
        """create_autoscaler works with TPU vm_type config."""
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
    min_slices: 0
    max_slices: 2
    zones: [us-central1-a]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        platform = create_platform(
            platform_config=config.platform,
            bootstrap_config=config.defaults.bootstrap,
            timeout_config=config.defaults.timeouts,
            ssh_config=config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=platform,
            autoscaler_config=config.defaults.autoscaler,
            scale_groups=config.scale_groups,
            dry_run=True,
        )

        assert autoscaler is not None
        assert "tpu_v5e_8" in autoscaler.groups

    def test_creates_autoscaler_with_manual_provider(self, tmp_path: Path):
        """create_autoscaler works with manual vm_type config."""
        config_content = """\
platform:
  manual: {}

defaults:
  bootstrap:
    docker_image: gcr.io/project/iris-worker:latest
    worker_port: 10001
    controller_address: "http://10.0.0.1:10000"
  ssh:
    user: ubuntu
    key_file: ~/.ssh/id_rsa

scale_groups:
  manual_hosts:
    vm_type: manual_vm
    accelerator_type: cpu
    manual:
      hosts: [10.0.0.1, 10.0.0.2]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        platform = create_platform(
            platform_config=config.platform,
            bootstrap_config=config.defaults.bootstrap,
            timeout_config=config.defaults.timeouts,
            ssh_config=config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=platform,
            autoscaler_config=config.defaults.autoscaler,
            scale_groups=config.scale_groups,
            dry_run=True,
        )

        assert autoscaler is not None
        assert "manual_hosts" in autoscaler.groups

    def test_creates_autoscaler_after_round_trip(self, tmp_path: Path):
        """create_autoscaler works after config round-trip."""
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
    min_slices: 0
    max_slices: 2
    zones: [us-central1-a]
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        # Load and round-trip
        original_config = load_config(config_path)
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Should be able to create autoscaler from round-tripped config
        platform = create_platform(
            platform_config=loaded_config.platform,
            bootstrap_config=loaded_config.defaults.bootstrap,
            timeout_config=loaded_config.defaults.timeouts,
            ssh_config=loaded_config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=platform,
            autoscaler_config=loaded_config.defaults.autoscaler,
            scale_groups=loaded_config.scale_groups,
            dry_run=True,
        )

        assert autoscaler is not None
        assert "tpu_v5e_8" in autoscaler.groups


class TestSshConfigMerging:
    """Tests for SSH config merging from cluster defaults and per-group overrides."""

    def test_uses_cluster_default_ssh_config(self):
        """get_ssh_config returns cluster defaults when no group override."""
        from iris.time_utils import Duration

        ssh_config_proto = config_pb2.SshConfig(
            user="ubuntu",
            key_file="~/.ssh/cluster_key",
            port=2222,
        )
        ssh_config_proto.connect_timeout.CopyFrom(Duration.from_seconds(60).to_proto())

        config = config_pb2.IrisClusterConfig(
            ssh=ssh_config_proto,
        )

        ssh_config = get_ssh_config(config)

        assert ssh_config.user == "ubuntu"
        assert ssh_config.key_file == "~/.ssh/cluster_key"
        assert ssh_config.port == 2222
        assert ssh_config.connect_timeout == Duration.from_seconds(60)

    def test_applies_per_group_ssh_overrides(self):
        """get_ssh_config applies per-group SSH overrides for manual vm_type."""
        config = config_pb2.IrisClusterConfig(
            ssh=config_pb2.SshConfig(
                user="ubuntu",
                key_file="~/.ssh/cluster_key",
                port=22,
            ),
            scale_groups={
                "manual_group": config_pb2.ScaleGroupConfig(
                    name="manual_group",
                    vm_type=config_pb2.VM_TYPE_MANUAL_VM,
                    manual=config_pb2.ManualProvider(
                        hosts=["10.0.0.1"],
                        ssh_user="admin",
                        ssh_key_file="~/.ssh/group_key",
                        ssh_port=2222,
                    ),
                )
            },
        )

        ssh_config = get_ssh_config(config, group_name="manual_group")

        assert ssh_config.user == "admin"
        assert ssh_config.key_file == "~/.ssh/group_key"
        assert ssh_config.port == 2222

    def test_partial_per_group_overrides_merge_with_defaults(self):
        """Per-group overrides merge with cluster defaults for unset fields."""
        from iris.time_utils import Duration

        ssh_config_proto = config_pb2.SshConfig(
            user="ubuntu",
            key_file="~/.ssh/cluster_key",
            port=22,
        )
        ssh_config_proto.connect_timeout.CopyFrom(Duration.from_seconds(30).to_proto())

        config = config_pb2.IrisClusterConfig(
            ssh=ssh_config_proto,
            scale_groups={
                "manual_group": config_pb2.ScaleGroupConfig(
                    name="manual_group",
                    vm_type=config_pb2.VM_TYPE_MANUAL_VM,
                    manual=config_pb2.ManualProvider(
                        hosts=["10.0.0.1"],
                        ssh_user="admin",  # Override user only
                    ),
                )
            },
        )

        ssh_config = get_ssh_config(config, group_name="manual_group")

        assert ssh_config.user == "admin"  # Overridden
        assert ssh_config.key_file == "~/.ssh/cluster_key"  # From default
        assert ssh_config.port == 22  # From default
        assert ssh_config.connect_timeout.to_seconds() == 30  # From default

    def test_uses_defaults_when_cluster_ssh_config_empty(self):
        """get_ssh_config uses built-in defaults when cluster config empty."""
        from iris.time_utils import Duration

        config = config_pb2.IrisClusterConfig()

        ssh_config = get_ssh_config(config)

        assert ssh_config.user == "root"
        assert ssh_config.key_file is None
        assert ssh_config.port == 22
        assert ssh_config.connect_timeout == Duration.from_seconds(30)
