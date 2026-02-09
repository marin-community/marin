# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for config loading, serialization, and deserialization.

These tests focus on stable behavior of config round-tripping through
YAML, ensuring that vm_type and platform configuration are preserved correctly.
"""

from pathlib import Path

import pytest
import yaml

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
    slice_size: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
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
    slice_size: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      tpu_count: 0
      gpu_count: 0
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

defaults:
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
    slice_size: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
  tpu_group_b:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    runtime_version: v2-alpha-tpuv5-lite
    slice_size: 16
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
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

    def test_example_eu_west4_config_round_trips(self, tmp_path: Path):
        """Real example config from examples/eu-west4.yaml round-trips correctly."""
        iris_root = Path(__file__).parent.parent.parent.parent
        config_path = iris_root / "examples" / "eu-west4.yaml"
        if not config_path.exists():
            pytest.skip("Example config not found")

        original_config = load_config(config_path)

        # Verify it has TPU vm_type before round-trip
        assert "tpu_v5e_16" in original_config.scale_groups
        assert original_config.scale_groups["tpu_v5e_16"].vm_type == config_pb2.VM_TYPE_TPU_VM

        # Round-trip via dict and YAML
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

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

defaults:
  bootstrap:
    docker_image: gcr.io/project/iris-worker:latest
    worker_port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  test_group:
    vm_type: manual_vm
    accelerator_type: {accelerator_type}
    slice_size: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      tpu_count: 0
      gpu_count: 0
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

defaults:
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
    slice_size: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
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
    slice_size: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
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
    slice_size: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      tpu_count: 0
      gpu_count: 0
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
    slice_size: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
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
        )
        ssh_config_proto.connect_timeout.CopyFrom(Duration.from_seconds(60).to_proto())

        config = config_pb2.IrisClusterConfig()
        config.defaults.ssh.CopyFrom(ssh_config_proto)

        ssh_config = get_ssh_config(config)

        assert ssh_config.user == "ubuntu"
        assert ssh_config.key_file == "~/.ssh/cluster_key"
        assert ssh_config.port == 22  # DEFAULT_SSH_PORT
        assert ssh_config.connect_timeout == Duration.from_seconds(60)

    def test_applies_per_group_ssh_overrides(self):
        """get_ssh_config applies per-group SSH overrides for manual vm_type."""
        config = config_pb2.IrisClusterConfig()
        config.defaults.ssh.user = "ubuntu"
        config.defaults.ssh.key_file = "~/.ssh/cluster_key"

        config.scale_groups["manual_group"].CopyFrom(
            config_pb2.ScaleGroupConfig(
                name="manual_group",
                vm_type=config_pb2.VM_TYPE_MANUAL_VM,
                manual=config_pb2.ManualProvider(
                    hosts=["10.0.0.1"],
                    ssh_user="admin",
                    ssh_key_file="~/.ssh/group_key",
                ),
            )
        )

        ssh_config = get_ssh_config(config, group_name="manual_group")

        assert ssh_config.user == "admin"
        assert ssh_config.key_file == "~/.ssh/group_key"
        assert ssh_config.port == 22

    def test_partial_per_group_overrides_merge_with_defaults(self):
        """Per-group overrides merge with cluster defaults for unset fields."""
        from iris.time_utils import Duration

        config = config_pb2.IrisClusterConfig()
        config.defaults.ssh.user = "ubuntu"
        config.defaults.ssh.key_file = "~/.ssh/cluster_key"
        config.defaults.ssh.connect_timeout.CopyFrom(Duration.from_seconds(30).to_proto())

        config.scale_groups["manual_group"].CopyFrom(
            config_pb2.ScaleGroupConfig(
                name="manual_group",
                vm_type=config_pb2.VM_TYPE_MANUAL_VM,
                manual=config_pb2.ManualProvider(
                    hosts=["10.0.0.1"],
                    ssh_user="admin",  # Override user only
                ),
            )
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


class TestLocalConfigTransformation:
    """Tests for make_local_config transformation."""

    def test_make_local_config_transforms_gcp_to_local(self, tmp_path: Path):
        """make_local_config transforms GCP config to local mode."""
        from iris.cluster.vm.config import make_local_config

        config_content = """\
platform:
  gcp:
    project_id: test-project
    zone: us-central1-a

defaults:
  bootstrap:
    docker_image: gcr.io/test/worker:latest
    worker_port: 10001
  autoscaler:
    evaluation_interval:
      milliseconds: 10000
    scale_up_delay:
      milliseconds: 60000
    scale_down_delay:
      milliseconds: 300000

controller:
  gcp:
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_group:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    slice_size: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
"""
        config_path = tmp_path / "gcp_config.yaml"
        config_path.write_text(config_content)

        # Load and transform
        original_config = load_config(config_path)
        local_config = make_local_config(original_config)

        # Verify platform transformed to local
        assert local_config.platform.WhichOneof("platform") == "local"

        # Verify controller transformed to local
        assert local_config.controller.WhichOneof("controller") == "local"
        assert local_config.controller.local.port == 0  # auto-assign

        # Verify scale groups transformed to local VM
        assert local_config.scale_groups["tpu_group"].vm_type == config_pb2.VM_TYPE_LOCAL_VM

        # Verify fast timings applied (0.5s eval, 1s scale_up)
        assert local_config.defaults.autoscaler.evaluation_interval.milliseconds == 500
        assert local_config.defaults.autoscaler.scale_up_delay.milliseconds == 1000
        # scale_down_delay stays at 5min
        assert local_config.defaults.autoscaler.scale_down_delay.milliseconds == 300000

    def test_make_local_config_preserves_scale_group_details(self, tmp_path: Path):
        """make_local_config preserves accelerator type and other scale group settings."""
        from iris.cluster.vm.config import make_local_config

        config_content = """\
platform:
  gcp:
    project_id: test-project
    zone: us-central1-a

defaults:
  bootstrap:
    docker_image: gcr.io/test/worker:latest

controller:
  gcp:
    port: 10000

scale_groups:
  cpu_group:
    vm_type: gce_vm
    accelerator_type: cpu
    slice_size: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      tpu_count: 0
      gpu_count: 0
    min_slices: 2
    max_slices: 5
    zones: [us-central1-a]
    priority: 50
  tpu_group:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    slice_size: 16
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 3
    zones: [us-central1-a]
    priority: 100
"""
        config_path = tmp_path / "multi_group.yaml"
        config_path.write_text(config_content)

        original_config = load_config(config_path)
        local_config = make_local_config(original_config)

        # Verify VM types changed but other fields preserved
        cpu_group = local_config.scale_groups["cpu_group"]
        assert cpu_group.vm_type == config_pb2.VM_TYPE_LOCAL_VM
        assert cpu_group.accelerator_type == config_pb2.ACCELERATOR_TYPE_CPU
        assert cpu_group.min_slices == 2
        assert cpu_group.max_slices == 5
        assert cpu_group.priority == 50

        tpu_group = local_config.scale_groups["tpu_group"]
        assert tpu_group.vm_type == config_pb2.VM_TYPE_LOCAL_VM
        assert tpu_group.accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU
        assert tpu_group.accelerator_variant == "v5litepod-16"
        assert tpu_group.min_slices == 1
        assert tpu_group.max_slices == 3
        assert tpu_group.priority == 100

    def test_example_configs_load_and_transform(self):
        """Example configs in examples/ directory load and transform to local correctly."""
        from iris.cluster.vm.config import make_local_config

        iris_root = Path(__file__).parent.parent.parent.parent
        example_configs = [
            iris_root / "examples" / "eu-west4.yaml",
            iris_root / "examples" / "demo.yaml",
        ]

        for config_path in example_configs:
            if not config_path.exists():
                pytest.skip(f"Example config not found: {config_path}")

            # Load the config
            config = load_config(config_path)
            assert config.platform.WhichOneof("platform") in ["gcp", "manual"]
            assert config.defaults.autoscaler.evaluation_interval.milliseconds > 0

            # Transform to local
            local_config = make_local_config(config)
            assert local_config.platform.WhichOneof("platform") == "local"
            assert local_config.controller.WhichOneof("controller") == "local"
            # Verify fast timings applied
            assert local_config.defaults.autoscaler.evaluation_interval.milliseconds == 500
            assert local_config.defaults.autoscaler.scale_up_delay.milliseconds == 1000
