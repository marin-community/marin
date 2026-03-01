# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for config loading, serialization, and deserialization.

These tests focus on stable behavior of config round-tripping through
YAML, ensuring that vm_type and platform configuration are preserved correctly.
"""

from pathlib import Path

import pytest
import yaml

from iris.cluster.config import (
    config_to_dict,
    create_autoscaler,
    get_ssh_config,
    load_config,
    validate_config,
)
from iris.cluster.platform.factory import create_platform
from iris.cluster.types import PREEMPTIBLE_ATTRIBUTE_KEY, REGION_ATTRIBUTE_KEY, ZONE_ATTRIBUTE_KEY
from iris.rpc import config_pb2


class TestConfigRoundTrip:
    """Tests for config serialization/deserialization round-trips."""

    def test_tpu_provider_survives_round_trip(self, tmp_path: Path):
        """TPU config survives proto→dict→yaml→dict→proto round-trip."""
        config_content = """\
platform:
  gcp:
    project_id: my-project

defaults:
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 10
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        # Load config from YAML
        original_config = load_config(config_path)

        # Verify accelerator type
        assert original_config.scale_groups["tpu_v5e_8"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify accelerator type is still TPU after round-trip
        assert loaded_config.scale_groups["tpu_v5e_8"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU

    def test_manual_provider_survives_round_trip(self, tmp_path: Path):
        """Manual config survives proto→dict→yaml→dict→proto round-trip."""
        config_content = """\
platform:
  manual: {}

defaults:
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"
  ssh:
    user: ubuntu
    key_file: ~/.ssh/id_rsa

scale_groups:
  manual_hosts:
    accelerator_type: cpu
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      tpu_count: 0
      gpu_count: 0
    slice_template:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
        ssh_user: ubuntu
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        # Load config from YAML
        original_config = load_config(config_path)

        # Verify manual hosts configuration
        assert original_config.scale_groups["manual_hosts"].HasField("slice_template")
        assert original_config.scale_groups["manual_hosts"].slice_template.HasField("manual")
        assert list(original_config.scale_groups["manual_hosts"].slice_template.manual.hosts) == [
            "10.0.0.1",
            "10.0.0.2",
        ]

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify manual hosts configuration survives round-trip
        assert loaded_config.scale_groups["manual_hosts"].HasField("slice_template")
        assert loaded_config.scale_groups["manual_hosts"].slice_template.HasField("manual")
        assert list(loaded_config.scale_groups["manual_hosts"].slice_template.manual.hosts) == [
            "10.0.0.1",
            "10.0.0.2",
        ]

    def test_multiple_scale_groups_preserve_accelerator_types(self, tmp_path: Path):
        """Config with multiple TPU scale groups preserves accelerator types."""
        config_content = """\
platform:
  gcp:
    project_id: my-project

defaults:
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_group_a:
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 10
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
  tpu_group_b:
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    num_vms: 16
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 0
    max_slices: 4
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
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

        assert loaded_config.scale_groups["tpu_group_a"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU
        assert loaded_config.scale_groups["tpu_group_b"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU

    def test_example_eu_west4_config_round_trips(self, tmp_path: Path):
        """Real example config from examples/marin.yaml round-trips correctly."""
        iris_root = Path(__file__).parent.parent.parent.parent
        config_path = iris_root / "examples" / "marin.yaml"
        if not config_path.exists():
            pytest.skip("Example config not found")

        original_config = load_config(config_path)

        # After expansion, zone-specific groups should exist
        assert "tpu_v5e_16-europe-west4-b" in original_config.scale_groups
        assert (
            original_config.scale_groups["tpu_v5e_16-europe-west4-b"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU
        )

        # Round-trip via dict and YAML
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        assert (
            loaded_config.scale_groups["tpu_v5e_16-europe-west4-b"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU
        )

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
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  test_group:
    accelerator_type: {accelerator_type}
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      tpu_count: 0
      gpu_count: 0
    slice_template:
      manual:
        hosts: [10.0.0.1]
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

defaults:
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_group:
    accelerator_type: TPU
    accelerator_variant: v5litepod-8
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 10
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.scale_groups["tpu_group"].accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU


class TestCreateAutoscalerFromConfig:
    """Tests for create_autoscaler factory function."""

    def test_creates_autoscaler_with_tpu_provider(self, tmp_path: Path):
        """create_autoscaler works with TPU config."""
        config_content = """\
platform:
  gcp:
    project_id: my-project

defaults:
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 0
    max_slices: 2
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        platform = create_platform(
            platform_config=config.platform,
            ssh_config=config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=platform,
            autoscaler_config=config.defaults.autoscaler,
            scale_groups=config.scale_groups,
            label_prefix=config.platform.label_prefix or "iris",
        )

        assert autoscaler is not None
        assert "tpu_v5e_8" in autoscaler.groups

    def test_creates_autoscaler_with_manual_provider(self, tmp_path: Path):
        """create_autoscaler works with manual config."""
        config_content = """\
platform:
  manual: {}

defaults:
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"
  ssh:
    user: ubuntu
    key_file: ~/.ssh/id_rsa

scale_groups:
  manual_hosts:
    accelerator_type: cpu
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      tpu_count: 0
      gpu_count: 0
    slice_template:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        platform = create_platform(
            platform_config=config.platform,
            ssh_config=config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=platform,
            autoscaler_config=config.defaults.autoscaler,
            scale_groups=config.scale_groups,
            label_prefix=config.platform.label_prefix or "iris",
        )

        assert autoscaler is not None
        assert "manual_hosts" in autoscaler.groups

    def test_creates_autoscaler_after_round_trip(self, tmp_path: Path):
        """create_autoscaler works after config round-trip."""
        config_content = """\
platform:
  gcp:
    project_id: my-project

defaults:
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_v5e_8:
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 0
    max_slices: 2
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
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
            ssh_config=loaded_config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=platform,
            autoscaler_config=loaded_config.defaults.autoscaler,
            scale_groups=loaded_config.scale_groups,
            label_prefix=loaded_config.platform.label_prefix or "iris",
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
        assert ssh_config.connect_timeout.milliseconds == 60_000

    def test_applies_per_group_ssh_overrides(self):
        """get_ssh_config applies per-group SSH overrides for manual slice template."""
        config = config_pb2.IrisClusterConfig()
        config.defaults.ssh.user = "ubuntu"
        config.defaults.ssh.key_file = "~/.ssh/cluster_key"

        manual_config = config_pb2.ScaleGroupConfig(
            name="manual_group",
        )
        manual_config.slice_template.manual.hosts.append("10.0.0.1")
        manual_config.slice_template.manual.ssh_user = "admin"
        manual_config.slice_template.manual.ssh_key_file = "~/.ssh/group_key"

        config.scale_groups["manual_group"].CopyFrom(manual_config)

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

        manual_config = config_pb2.ScaleGroupConfig(
            name="manual_group",
        )
        manual_config.slice_template.manual.hosts.append("10.0.0.1")
        manual_config.slice_template.manual.ssh_user = "admin"  # Override user only

        config.scale_groups["manual_group"].CopyFrom(manual_config)

        ssh_config = get_ssh_config(config, group_name="manual_group")

        assert ssh_config.user == "admin"  # Overridden
        assert ssh_config.key_file == "~/.ssh/cluster_key"  # From default
        assert ssh_config.port == 22  # From default
        assert ssh_config.connect_timeout.milliseconds == 30_000  # From default

    def test_uses_defaults_when_cluster_ssh_config_empty(self):
        """get_ssh_config uses built-in defaults when cluster config empty."""

        config = config_pb2.IrisClusterConfig()

        ssh_config = get_ssh_config(config)

        assert ssh_config.user == "root"
        assert ssh_config.key_file == ""
        assert ssh_config.port == 22
        assert ssh_config.connect_timeout.milliseconds == 30_000


class TestLocalConfigTransformation:
    """Tests for make_local_config transformation."""

    def test_make_local_config_transforms_gcp_to_local(self, tmp_path: Path):
        """make_local_config transforms GCP config to local mode."""
        from iris.cluster.config import make_local_config

        config_content = """\
platform:
  gcp:
    project_id: test-project

defaults:
  worker:
    docker_image: gcr.io/test/worker:latest
    port: 10001
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
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 10
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: tpu-ubuntu2204-base
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

        # Verify fast timings applied (0.5s eval, 1s scale_up)
        assert local_config.defaults.autoscaler.evaluation_interval.milliseconds == 500
        assert local_config.defaults.autoscaler.scale_up_delay.milliseconds == 1000
        # scale_down_delay stays at 5min
        assert local_config.defaults.autoscaler.scale_down_delay.milliseconds == 300000

    def test_make_local_config_preserves_scale_group_details(self, tmp_path: Path):
        """make_local_config preserves accelerator type and other scale group settings."""
        from iris.cluster.config import make_local_config

        config_content = """\
platform:
  gcp:
    project_id: test-project

defaults:
  worker:
    docker_image: gcr.io/test/worker:latest

controller:
  gcp:
    port: 10000

scale_groups:
  cpu_group:
    accelerator_type: cpu
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      tpu_count: 0
      gpu_count: 0
    min_slices: 2
    max_slices: 5
    priority: 50
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: cos-stable
  tpu_group:
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    num_vms: 16
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      tpu_count: 8
      gpu_count: 0
    min_slices: 1
    max_slices: 3
    priority: 100
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: tpu-ubuntu2204-base
"""
        config_path = tmp_path / "multi_group.yaml"
        config_path.write_text(config_content)

        original_config = load_config(config_path)
        local_config = make_local_config(original_config)

        # Verify other fields preserved
        cpu_group = local_config.scale_groups["cpu_group"]
        assert cpu_group.accelerator_type == config_pb2.ACCELERATOR_TYPE_CPU
        assert cpu_group.min_slices == 2
        assert cpu_group.max_slices == 5
        assert cpu_group.priority == 50

        tpu_group = local_config.scale_groups["tpu_group"]
        assert tpu_group.accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU
        assert tpu_group.accelerator_variant == "v5litepod-16"
        assert tpu_group.min_slices == 1
        assert tpu_group.max_slices == 3
        assert tpu_group.priority == 100

    def test_example_configs_load_and_transform(self):
        """Example configs in examples/ directory load and transform to local correctly."""
        from iris.cluster.config import make_local_config

        iris_root = Path(__file__).parent.parent.parent.parent
        example_configs = [
            iris_root / "examples" / "marin.yaml",
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


def _valid_scale_group() -> config_pb2.ScaleGroupConfig:
    """Create a valid ScaleGroupConfig for use in validation tests."""
    sg = config_pb2.ScaleGroupConfig(
        name="test",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3),
    )
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.num_vms = 1
    sg.slice_template.local.SetInParent()
    return sg


def _config_with(**overrides) -> config_pb2.IrisClusterConfig:
    """Build an IrisClusterConfig with a single scale group, overriding fields."""
    sg = _valid_scale_group()
    for key, value in overrides.items():
        if key == "resources":
            sg.resources.CopyFrom(value)
        else:
            setattr(sg, key, value)
    config = config_pb2.IrisClusterConfig()
    config.scale_groups["test"].CopyFrom(sg)
    return config


class TestConfigValidation:
    """Tests for validate_config: the consolidated entry point for config validation."""

    def test_valid_config_accepted(self):
        validate_config(_config_with())

    def test_rejects_missing_resources(self):
        config = config_pb2.IrisClusterConfig()
        sg = config.scale_groups["test"]
        sg.name = "test"
        sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.num_vms = 1
        with pytest.raises(ValueError, match="must set resources"):
            validate_config(config)

    def test_rejects_missing_num_vms(self):
        config = config_pb2.IrisClusterConfig()
        sg = config.scale_groups["test"]
        sg.name = "test"
        sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.resources.CopyFrom(config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3))
        with pytest.raises(ValueError, match="must set num_vms"):
            validate_config(config)

    def test_rejects_zero_num_vms(self):
        with pytest.raises(ValueError, match="invalid num_vms"):
            validate_config(_config_with(num_vms=0))

    def test_rejects_unspecified_accelerator_type(self):
        with pytest.raises(ValueError, match="must set accelerator_type"):
            validate_config(_config_with(accelerator_type=config_pb2.ACCELERATOR_TYPE_UNSPECIFIED))

    def test_rejects_negative_cpu(self):
        with pytest.raises(ValueError, match="invalid cpu_millicores"):
            validate_config(
                _config_with(resources=config_pb2.ScaleGroupResources(cpu_millicores=-1000, memory_bytes=16 * 1024**3))
            )

    def test_rejects_gcp_zone_not_in_platform_zones(self):
        """Validation fails when scale group zone is not in platform.gcp.zones."""
        config = config_pb2.IrisClusterConfig()
        config.platform.gcp.project_id = "test"
        config.platform.gcp.zones.append("zone-a")

        sg = config_pb2.ScaleGroupConfig(
            name="tpu",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-8",
            num_vms=8,
            resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3, tpu_count=4),
        )
        sg.slice_template.gcp.zone = "zone-b"
        sg.slice_template.gcp.runtime_version = "v2-alpha-tpuv5-lite"
        config.scale_groups["tpu"].CopyFrom(sg)

        with pytest.raises(ValueError, match=r"not in platform\.gcp\.zones"):
            validate_config(config)

    def test_accepts_gcp_zone_in_platform_zones(self):
        """Validation passes when scale group zone is in platform.gcp.zones."""
        config = config_pb2.IrisClusterConfig()
        config.platform.gcp.project_id = "test"
        config.platform.gcp.zones.append("zone-a")

        sg = config_pb2.ScaleGroupConfig(
            name="tpu",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-8",
            num_vms=8,
            resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3, tpu_count=4),
        )
        sg.slice_template.gcp.zone = "zone-a"
        sg.slice_template.gcp.runtime_version = "v2-alpha-tpuv5-lite"
        config.scale_groups["tpu"].CopyFrom(sg)

        validate_config(config)  # Should not raise

    def test_rejects_gcp_vm_mode_with_preemptible(self):
        config = config_pb2.IrisClusterConfig()
        sg = config_pb2.ScaleGroupConfig(
            name="cpu-vm",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
            num_vms=1,
            resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3),
        )
        sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.slice_template.preemptible = True
        sg.slice_template.gcp.zone = "us-central1-a"
        sg.slice_template.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
        sg.slice_template.gcp.machine_type = "n2-standard-4"
        config.scale_groups["cpu-vm"].CopyFrom(sg)

        with pytest.raises(ValueError, match="do not support preemptible"):
            validate_config(config)

    def test_rejects_gcp_vm_mode_with_num_vms_not_one(self):
        config = config_pb2.IrisClusterConfig()
        sg = config_pb2.ScaleGroupConfig(
            name="cpu-vm",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
            num_vms=2,
            resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3),
        )
        sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.slice_template.gcp.zone = "us-central1-a"
        sg.slice_template.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
        sg.slice_template.gcp.machine_type = "n2-standard-4"
        config.scale_groups["cpu-vm"].CopyFrom(sg)

        with pytest.raises(ValueError, match="require num_vms=1"):
            validate_config(config)

    def test_rejects_gcp_vm_mode_with_non_cpu_accelerator(self):
        config = config_pb2.IrisClusterConfig()
        sg = config_pb2.ScaleGroupConfig(
            name="gpu-vm",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
            num_vms=1,
            resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3, gpu_count=1),
        )
        sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_GPU
        sg.slice_template.gcp.zone = "us-central1-a"
        sg.slice_template.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
        sg.slice_template.gcp.machine_type = "n2-standard-4"
        config.scale_groups["gpu-vm"].CopyFrom(sg)

        with pytest.raises(ValueError, match="require accelerator_type=cpu"):
            validate_config(config)

    def test_accepts_gcp_vm_mode_cpu_single_vm_non_preemptible(self):
        config = config_pb2.IrisClusterConfig()
        sg = config_pb2.ScaleGroupConfig(
            name="cpu-vm",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
            num_vms=1,
            resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3),
        )
        sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.slice_template.gcp.zone = "us-central1-a"
        sg.slice_template.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
        sg.slice_template.gcp.machine_type = "n2-standard-4"
        config.scale_groups["cpu-vm"].CopyFrom(sg)

        validate_config(config)


def _gcp_scale_group(zone: str, *, preemptible: bool = False) -> config_pb2.ScaleGroupConfig:
    """Build a valid GCP-backed ScaleGroupConfig for worker settings validation tests."""
    sg = config_pb2.ScaleGroupConfig(
        name="test",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(cpu_millicores=8000, memory_bytes=16 * 1024**3, tpu_count=1),
    )
    sg.slice_template.gcp.zone = zone
    sg.slice_template.gcp.runtime_version = "v2-alpha-tpuv5-lite"
    sg.slice_template.preemptible = preemptible
    return sg


def _config_with_gcp_sg(
    zone: str,
    *,
    preemptible: bool = False,
    worker_attributes: dict[str, str] | None = None,
) -> config_pb2.IrisClusterConfig:
    """Build an IrisClusterConfig containing a single GCP scale group with optional worker attributes."""
    sg = _gcp_scale_group(zone, preemptible=preemptible)
    if worker_attributes is not None:
        for k, v in worker_attributes.items():
            sg.worker.attributes[k] = v
    config = config_pb2.IrisClusterConfig()
    config.scale_groups["test"].CopyFrom(sg)
    return config


class TestWorkerSettingsValidation:
    """Tests for worker.attributes validation (region/zone consistency, preemptible agreement)."""

    def test_no_worker_settings_accepted(self):
        """Scale groups without worker settings always pass validation."""
        config = _config_with_gcp_sg("us-west4-b")
        validate_config(config)

    def test_region_matching_zone_prefix_accepted(self):
        """worker.attributes.region that matches zone prefix passes validation."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={REGION_ATTRIBUTE_KEY: "us-west4"})
        validate_config(config)

    @pytest.mark.parametrize(
        "zone,region",
        [
            ("us-west4-b", "us-central1"),
            ("us-central1-a", "us-west4"),
            ("europe-west4-a", "us-west4"),
        ],
    )
    def test_region_mismatching_zone_prefix_rejected(self, zone: str, region: str):
        """worker.attributes.region that doesn't match zone prefix fails validation."""
        config = _config_with_gcp_sg(zone, worker_attributes={REGION_ATTRIBUTE_KEY: region})
        with pytest.raises(ValueError, match="must match"):
            validate_config(config)

    def test_empty_region_attribute_rejected(self):
        """worker.attributes.region set to empty string fails validation."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={REGION_ATTRIBUTE_KEY: ""})
        with pytest.raises(ValueError, match="must be non-empty"):
            validate_config(config)

    def test_preemptible_true_matches_template_true(self):
        """worker.attributes.preemptible='true' with slice_template.preemptible=True passes."""
        config = _config_with_gcp_sg(
            "us-west4-b", preemptible=True, worker_attributes={PREEMPTIBLE_ATTRIBUTE_KEY: "true"}
        )
        validate_config(config)

    def test_preemptible_false_matches_template_false(self):
        """worker.attributes.preemptible='false' with slice_template.preemptible=False passes."""
        config = _config_with_gcp_sg(
            "us-west4-b", preemptible=False, worker_attributes={PREEMPTIBLE_ATTRIBUTE_KEY: "false"}
        )
        validate_config(config)

    @pytest.mark.parametrize(
        "attr_value,template_preemptible",
        [
            ("true", False),
            ("false", True),
        ],
    )
    def test_preemptible_mismatch_rejected(self, attr_value: str, template_preemptible: bool):
        """worker.attributes.preemptible disagreeing with slice_template.preemptible fails."""
        config = _config_with_gcp_sg(
            "us-west4-b",
            preemptible=template_preemptible,
            worker_attributes={PREEMPTIBLE_ATTRIBUTE_KEY: attr_value},
        )
        with pytest.raises(ValueError, match="must match"):
            validate_config(config)

    def test_invalid_preemptible_attribute_value_rejected(self):
        """worker.attributes.preemptible with a non-boolean string fails validation."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={PREEMPTIBLE_ATTRIBUTE_KEY: "yes"})
        with pytest.raises(ValueError, match="must be 'true' or 'false'"):
            validate_config(config)

    def test_both_region_and_preemptible_valid_together(self):
        """Both region and preemptible worker attributes that agree with slice_template pass."""
        config = _config_with_gcp_sg(
            "us-west4-b",
            preemptible=True,
            worker_attributes={REGION_ATTRIBUTE_KEY: "us-west4", PREEMPTIBLE_ATTRIBUTE_KEY: "true"},
        )
        validate_config(config)

    def test_region_check_skipped_for_non_gcp_platform(self):
        """Region/zone consistency is only checked for GCP slice templates; manual/local groups are unaffected."""
        sg = _valid_scale_group()  # uses local platform
        sg.worker.attributes[REGION_ATTRIBUTE_KEY] = "us-west4"
        config = config_pb2.IrisClusterConfig()
        config.scale_groups["test"].CopyFrom(sg)
        validate_config(config)  # no GCP zone — region check does not apply

    def test_zone_matching_gcp_zone_accepted(self):
        """worker.attributes.zone matching slice_template.gcp.zone passes validation."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={ZONE_ATTRIBUTE_KEY: "us-west4-b"})
        validate_config(config)

    def test_zone_mismatching_gcp_zone_rejected(self):
        """worker.attributes.zone not matching slice_template.gcp.zone fails validation."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={ZONE_ATTRIBUTE_KEY: "us-west4-a"})
        with pytest.raises(ValueError, match="must match"):
            validate_config(config)

    def test_empty_zone_attribute_rejected(self):
        """worker.attributes.zone set to empty string fails validation."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={ZONE_ATTRIBUTE_KEY: ""})
        with pytest.raises(ValueError, match="must be non-empty"):
            validate_config(config)


class TestMultiZoneExpansion:
    """Tests for zones-based scale group expansion."""

    def test_expands_into_per_zone_groups(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

scale_groups:
  tpu_v5e_16:
    zones: [europe-west4-b, us-west4-a]
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    num_vms: 4
    resources: { cpu: 128, ram: 128GB, disk: 1TB, tpu_count: 4, gpu_count: 0 }
    max_slices: 4
    slice_template:
      preemptible: true
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
    worker:
      attributes:
        preemptible: "true"
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)

        assert "tpu_v5e_16" not in config.scale_groups
        assert "tpu_v5e_16-europe-west4-b" in config.scale_groups
        assert "tpu_v5e_16-us-west4-a" in config.scale_groups

        eu = config.scale_groups["tpu_v5e_16-europe-west4-b"]
        assert eu.slice_template.gcp.zone == "europe-west4-b"
        assert eu.worker.attributes["zone"] == "europe-west4-b"
        assert eu.worker.attributes["region"] == "europe-west4"
        assert eu.min_slices == 0

        us = config.scale_groups["tpu_v5e_16-us-west4-a"]
        assert us.slice_template.gcp.zone == "us-west4-a"
        assert us.worker.attributes["zone"] == "us-west4-a"
        assert us.worker.attributes["region"] == "us-west4"

    def test_min_slices_preserved_when_explicit(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

scale_groups:
  tpu_group:
    zones: [us-west4-a]
    accelerator_type: tpu
    num_vms: 1
    min_slices: 2
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)
        assert config.scale_groups["tpu_group-us-west4-a"].min_slices == 2

    def test_groups_without_zones_unchanged(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test
    zones: [us-west4-a]

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

scale_groups:
  static_group:
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    min_slices: 1
    slice_template:
      gcp:
        zone: us-west4-a
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)
        assert "static_group" in config.scale_groups
        assert config.scale_groups["static_group"].min_slices == 1

    def test_zones_auto_populated_in_platform(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

scale_groups:
  tpu_group:
    zones: [us-west4-a, europe-west4-b]
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)
        zones = set(config.platform.gcp.zones)
        assert "us-west4-a" in zones
        assert "europe-west4-b" in zones

    def test_empty_zones_list_rejected(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  tpu_group:
    zones: []
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="non-empty"):
            load_config(p)

    def test_mixed_expanded_and_static_groups(self, tmp_path: Path):
        """Expanded and non-expanded groups coexist."""
        config_content = """\
platform:
  gcp:
    project_id: test
    zones: [us-central1-a]

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

scale_groups:
  static_cpu:
    accelerator_type: cpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 0, gpu_count: 0 }
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: cos-stable
  expanded_tpu:
    zones: [us-west4-a, europe-west4-b]
    accelerator_type: tpu
    num_vms: 4
    resources: { cpu: 128, ram: 128GB, disk: 1TB, tpu_count: 4, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)
        assert "static_cpu" in config.scale_groups
        assert "expanded_tpu-us-west4-a" in config.scale_groups
        assert "expanded_tpu-europe-west4-b" in config.scale_groups
        assert "expanded_tpu" not in config.scale_groups

    def test_duplicate_zones_rejected(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  tpu_group:
    zones: [us-west4-a, us-west4-a]
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="duplicates"):
            load_config(p)

    def test_non_string_zone_rejected(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  tpu_group:
    zones: [123]
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="non-empty string"):
            load_config(p)

    def test_conflicting_gcp_zone_rejected(self, tmp_path: Path):
        """User-provided slice_template.gcp.zone conflicts with zones expansion."""
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  tpu_group:
    zones: [us-west4-a]
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        zone: europe-west4-b
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="cannot set both"):
            load_config(p)

    def test_conflicting_worker_zone_attr_rejected(self, tmp_path: Path):
        """User-provided worker.attributes.zone conflicts with zones expansion."""
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  tpu_group:
    zones: [us-west4-a]
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
    worker:
      attributes:
        zone: "us-west4-a"
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="cannot set both"):
            load_config(p)

    def test_conflicting_worker_region_attr_rejected(self, tmp_path: Path):
        """User-provided worker.attributes.region conflicts with zones expansion."""
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  tpu_group:
    zones: [us-west4-a]
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
    worker:
      attributes:
        region: "us-west4"
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="cannot set both"):
            load_config(p)

    def test_non_gcp_slice_template_rejected(self, tmp_path: Path):
        """Zone expansion on a non-GCP slice template is rejected."""
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  manual_group:
    zones: [us-west4-a]
    accelerator_type: cpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 0, gpu_count: 0 }
    slice_template:
      manual:
        hosts: [10.0.0.1]
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="only supported for GCP"):
            load_config(p)

    def test_name_collision_with_existing_group_rejected(self, tmp_path: Path):
        """Expanded name colliding with an existing static group is rejected."""
        config_content = """\
platform:
  gcp:
    project_id: test
    zones: [us-west4-a]

scale_groups:
  tpu_group-us-west4-a:
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        zone: us-west4-a
        runtime_version: v2-alpha-tpuv5-lite
  tpu_group:
    zones: [us-west4-a]
    accelerator_type: tpu
    num_vms: 1
    resources: { cpu: 8, ram: 16GB, disk: 50GB, tpu_count: 1, gpu_count: 0 }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="collides"):
            load_config(p)
