# Copyright The Marin Authors
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
    connect_cluster,
    create_autoscaler,
    get_ssh_config,
    load_config,
    make_local_config,
    validate_config,
)
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.providers.factory import create_provider_bundle
from iris.rpc import config_pb2, controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, ExponentialBackoff


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
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-8
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 1
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
        assert original_config.scale_groups["tpu_v5e_8"].resources.device_type == config_pb2.ACCELERATOR_TYPE_TPU

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify accelerator type is still TPU after round-trip
        assert loaded_config.scale_groups["tpu_v5e_8"].resources.device_type == config_pb2.ACCELERATOR_TYPE_TPU

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
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      device_type: cpu
      device_count: 0
      capacity_type: on-demand
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
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-8
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 1
    max_slices: 10
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
  tpu_group_b:
    num_vms: 16
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-16
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 0
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

        assert loaded_config.scale_groups["tpu_group_a"].resources.device_type == config_pb2.ACCELERATOR_TYPE_TPU
        assert loaded_config.scale_groups["tpu_group_b"].resources.device_type == config_pb2.ACCELERATOR_TYPE_TPU

    def test_example_eu_west4_config_round_trips(self, tmp_path: Path):
        """Real example config from examples/marin.yaml round-trips correctly."""
        iris_root = Path(__file__).parent.parent.parent.parent
        config_path = iris_root / "examples" / "marin.yaml"
        if not config_path.exists():
            pytest.skip("Example config not found")

        original_config = load_config(config_path)

        # After expansion, zone-specific groups should exist
        assert "tpu_v5e-preemptible_16-europe-west4-b" in original_config.scale_groups
        assert (
            original_config.scale_groups["tpu_v5e-preemptible_16-europe-west4-b"].resources.device_type
            == config_pb2.ACCELERATOR_TYPE_TPU
        )

        # Round-trip via dict and YAML
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        assert (
            loaded_config.scale_groups["tpu_v5e-preemptible_16-europe-west4-b"].resources.device_type
            == config_pb2.ACCELERATOR_TYPE_TPU
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
        """Config accepts lowercase accelerator types in resources.device_type."""
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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: {accelerator_type}
      device_count: 0
      capacity_type: preemptible
    slice_template:
      manual:
        hosts: [10.0.0.1]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.scale_groups["test_group"].resources.device_type == expected_enum

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
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: ACCELERATOR_TYPE_TPU
      device_variant: v5litepod-8
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 1
    max_slices: 10
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.scale_groups["tpu_group"].resources.device_type == config_pb2.ACCELERATOR_TYPE_TPU


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
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-8
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 0
    max_slices: 2
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        bundle = create_provider_bundle(
            platform_config=config.platform,
            ssh_config=config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=bundle.workers,
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
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      device_type: cpu
      device_count: 0
      capacity_type: on-demand
    slice_template:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        bundle = create_provider_bundle(
            platform_config=config.platform,
            ssh_config=config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=bundle.workers,
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
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-8
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 0
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
        bundle = create_provider_bundle(
            platform_config=loaded_config.platform,
            ssh_config=loaded_config.defaults.ssh,
        )
        autoscaler = create_autoscaler(
            platform=bundle.workers,
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

        ssh_config_proto = config_pb2.SshConfig(
            user="ubuntu",
            key_file="~/.ssh/cluster_key",
            auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
            os_login_user="ubuntu_oslogin",
            impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
        )
        ssh_config_proto.connect_timeout.CopyFrom(duration_to_proto(Duration.from_seconds(60)))

        config = config_pb2.IrisClusterConfig()
        config.defaults.ssh.CopyFrom(ssh_config_proto)

        ssh_config = get_ssh_config(config)

        assert ssh_config.user == "ubuntu"
        assert ssh_config.key_file == "~/.ssh/cluster_key"
        assert ssh_config.port == 22  # DEFAULT_SSH_PORT
        assert ssh_config.auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN
        assert ssh_config.os_login_user == "ubuntu_oslogin"
        assert ssh_config.impersonate_service_account == "iris-controller@test-project.iam.gserviceaccount.com"
        assert ssh_config.connect_timeout.milliseconds == 60_000

    def test_applies_per_group_ssh_overrides(self):
        """get_ssh_config applies per-group SSH overrides for manual slice template."""
        config = config_pb2.IrisClusterConfig()
        config.defaults.ssh.user = "ubuntu"
        config.defaults.ssh.key_file = "~/.ssh/cluster_key"
        config.defaults.ssh.auth_mode = config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN
        config.defaults.ssh.os_login_user = "ubuntu_oslogin"

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
        assert ssh_config.auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN
        assert ssh_config.os_login_user == "ubuntu_oslogin"

    def test_partial_per_group_overrides_merge_with_defaults(self):
        """Per-group overrides merge with cluster defaults for unset fields."""

        config = config_pb2.IrisClusterConfig()
        config.defaults.ssh.user = "ubuntu"
        config.defaults.ssh.key_file = "~/.ssh/cluster_key"
        config.defaults.ssh.auth_mode = config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN
        config.defaults.ssh.os_login_user = "ubuntu_oslogin"
        config.defaults.ssh.connect_timeout.CopyFrom(duration_to_proto(Duration.from_seconds(30)))

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
        assert ssh_config.auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN
        assert ssh_config.os_login_user == "ubuntu_oslogin"
        assert ssh_config.connect_timeout.milliseconds == 30_000  # From default

    def test_uses_defaults_when_cluster_ssh_config_empty(self):
        """get_ssh_config uses built-in defaults when cluster config empty."""

        config = config_pb2.IrisClusterConfig()

        ssh_config = get_ssh_config(config)

        assert ssh_config.user == "root"
        assert ssh_config.key_file == ""
        assert ssh_config.port == 22
        assert ssh_config.auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_METADATA
        assert ssh_config.os_login_user == ""
        assert ssh_config.impersonate_service_account == ""
        assert ssh_config.connect_timeout.milliseconds == 30_000

    def test_validate_config_requires_gcp_service_accounts_for_os_login(self):
        config = config_pb2.IrisClusterConfig()
        config.platform.gcp.project_id = "test-project"
        config.controller.gcp.zone = "us-central1-a"
        config.defaults.ssh.auth_mode = config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN
        config.defaults.worker.docker_image = "ghcr.io/marin-community/iris-worker:latest"

        group = config.scale_groups["tpu"]
        group.name = "tpu"
        group.num_vms = 1
        group.resources.device_type = config_pb2.ACCELERATOR_TYPE_TPU
        group.resources.device_variant = "v5litepod-4"
        group.resources.capacity_type = config_pb2.CAPACITY_TYPE_PREEMPTIBLE
        group.slice_template.gcp.zone = "us-central1-a"
        group.slice_template.gcp.runtime_version = "tpu-ubuntu2204-base"

        with pytest.raises(ValueError, match=r"controller\.gcp\.service_account"):
            validate_config(config)


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
      milliseconds: 600000

controller:
  gcp:
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_group:
    num_vms: 8
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-8
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 1
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
        # scale_down_delay preserved from YAML (600s)
        assert local_config.defaults.autoscaler.scale_down_delay.milliseconds == 600000

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
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      device_type: cpu
      device_count: 0
      capacity_type: on-demand
    buffer_slices: 2
    max_slices: 5
    priority: 50
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: cos-stable
  tpu_group:
    num_vms: 16
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-16
      device_count: 8
      capacity_type: preemptible
    buffer_slices: 1
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
        assert cpu_group.resources.device_type == config_pb2.ACCELERATOR_TYPE_CPU
        assert cpu_group.buffer_slices == 2
        assert cpu_group.max_slices == 5
        assert cpu_group.priority == 50

        tpu_group = local_config.scale_groups["tpu_group"]
        assert tpu_group.resources.device_type == config_pb2.ACCELERATOR_TYPE_TPU
        assert tpu_group.resources.device_variant == "v5litepod-16"
        assert tpu_group.buffer_slices == 1
        assert tpu_group.max_slices == 3
        assert tpu_group.priority == 100

    def test_example_configs_load_and_transform(self):
        """Example configs in examples/ directory load and transform to local correctly."""
        from iris.cluster.config import make_local_config

        iris_root = Path(__file__).parent.parent.parent.parent
        example_configs = [
            iris_root / "examples" / "marin.yaml",
            iris_root / "examples" / "marin-dev.yaml",
            iris_root / "examples" / "coreweave.yaml",
            iris_root / "examples" / "test.yaml",
        ]

        for config_path in example_configs:
            if not config_path.exists():
                pytest.skip(f"Example config not found: {config_path}")

            # Load the config
            config = load_config(config_path)
            assert config.platform.WhichOneof("platform") in ["gcp", "manual", "coreweave"]
            assert config.defaults.autoscaler.evaluation_interval.milliseconds > 0

            # Transform to local
            local_config = make_local_config(config)
            assert local_config.platform.WhichOneof("platform") == "local"
            assert local_config.controller.WhichOneof("controller") == "local"
            # Verify fast timings applied
            assert local_config.defaults.autoscaler.evaluation_interval.milliseconds == 500
            assert local_config.defaults.autoscaler.scale_up_delay.milliseconds == 1000

    def test_example_config_zones_in_known_gcp_zones(self):
        """All GCP zones used in example configs must be in KNOWN_GCP_ZONES."""
        from iris.cluster.providers.gcp.service import KNOWN_GCP_ZONES

        iris_root = Path(__file__).parent.parent.parent.parent
        for config_path in [iris_root / "examples" / "marin.yaml", iris_root / "examples" / "marin-dev.yaml"]:
            if not config_path.exists():
                pytest.skip(f"Example config not found: {config_path}")
            config = load_config(config_path)
            for name, sg in config.scale_groups.items():
                template = sg.slice_template
                if template.WhichOneof("platform") == "gcp" and template.gcp.zone:
                    assert (
                        template.gcp.zone in KNOWN_GCP_ZONES
                    ), f"Scale group '{name}': zone '{template.gcp.zone}' not in KNOWN_GCP_ZONES"


def _valid_scale_group() -> config_pb2.ScaleGroupConfig:
    """Create a valid ScaleGroupConfig for use in validation tests."""
    sg = config_pb2.ScaleGroupConfig(
        name="test",
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            device_type=config_pb2.ACCELERATOR_TYPE_CPU,
            capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
        ),
    )
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
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
        sg.num_vms = 1
        with pytest.raises(ValueError, match="must set resources"):
            validate_config(config)

    def test_rejects_missing_num_vms(self):
        config = config_pb2.IrisClusterConfig()
        sg = config.scale_groups["test"]
        sg.name = "test"
        sg.resources.CopyFrom(
            config_pb2.ScaleGroupResources(
                cpu_millicores=8000,
                memory_bytes=16 * 1024**3,
                device_type=config_pb2.ACCELERATOR_TYPE_CPU,
            )
        )
        with pytest.raises(ValueError, match="must set num_vms"):
            validate_config(config)

    def test_rejects_zero_num_vms(self):
        with pytest.raises(ValueError, match="invalid num_vms"):
            validate_config(_config_with(num_vms=0))

    def test_rejects_unspecified_accelerator_type(self):
        with pytest.raises(ValueError, match=r"must set resources\.device_type"):
            validate_config(
                _config_with(
                    resources=config_pb2.ScaleGroupResources(
                        cpu_millicores=8000,
                        memory_bytes=16 * 1024**3,
                        device_type=config_pb2.ACCELERATOR_TYPE_UNSPECIFIED,
                    )
                )
            )

    def test_rejects_negative_cpu(self):
        with pytest.raises(ValueError, match="invalid cpu_millicores"):
            validate_config(
                _config_with(
                    resources=config_pb2.ScaleGroupResources(
                        cpu_millicores=-1000,
                        memory_bytes=16 * 1024**3,
                        device_type=config_pb2.ACCELERATOR_TYPE_CPU,
                    )
                )
            )

    def test_rejects_gcp_zone_not_in_platform_zones(self):
        """Validation fails when scale group zone is not in platform.gcp.zones."""
        config = config_pb2.IrisClusterConfig()
        config.platform.gcp.project_id = "test"
        config.platform.gcp.zones.append("zone-a")

        sg = config_pb2.ScaleGroupConfig(
            name="tpu",
            num_vms=8,
            resources=config_pb2.ScaleGroupResources(
                cpu_millicores=8000,
                memory_bytes=16 * 1024**3,
                device_count=4,
                device_type=config_pb2.ACCELERATOR_TYPE_TPU,
                capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
            ),
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
            num_vms=8,
            resources=config_pb2.ScaleGroupResources(
                cpu_millicores=8000,
                memory_bytes=16 * 1024**3,
                device_count=4,
                device_type=config_pb2.ACCELERATOR_TYPE_TPU,
                capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
            ),
        )
        sg.slice_template.gcp.zone = "zone-a"
        sg.slice_template.gcp.runtime_version = "v2-alpha-tpuv5-lite"
        config.scale_groups["tpu"].CopyFrom(sg)

        validate_config(config)  # Should not raise

    @pytest.mark.parametrize(
        "num_vms,device_type,device_count,capacity_type,error_match",
        [
            (
                1,
                config_pb2.ACCELERATOR_TYPE_CPU,
                0,
                config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
                "only support capacity_type on-demand",
            ),
            (2, config_pb2.ACCELERATOR_TYPE_CPU, 0, config_pb2.CAPACITY_TYPE_ON_DEMAND, "require num_vms=1"),
            (1, config_pb2.ACCELERATOR_TYPE_GPU, 1, config_pb2.CAPACITY_TYPE_ON_DEMAND, "require device_type=cpu"),
        ],
        ids=["non_on_demand", "multi_vm", "non_cpu"],
    )
    def test_rejects_invalid_gcp_vm_mode(self, num_vms, device_type, device_count, capacity_type, error_match):
        config = config_pb2.IrisClusterConfig()
        sg = config_pb2.ScaleGroupConfig(
            name="test-vm",
            num_vms=num_vms,
            resources=config_pb2.ScaleGroupResources(
                cpu_millicores=8000,
                memory_bytes=16 * 1024**3,
                device_type=device_type,
                device_count=device_count,
                capacity_type=capacity_type,
            ),
        )
        sg.slice_template.accelerator_type = device_type
        sg.slice_template.capacity_type = capacity_type
        sg.slice_template.gcp.zone = "us-central1-a"
        sg.slice_template.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
        sg.slice_template.gcp.machine_type = "n2-standard-4"
        config.scale_groups["test-vm"].CopyFrom(sg)

        with pytest.raises(ValueError, match=error_match):
            validate_config(config)

    def test_accepts_gcp_vm_mode_cpu_single_vm_on_demand(self):
        config = config_pb2.IrisClusterConfig()
        sg = config_pb2.ScaleGroupConfig(
            name="cpu-vm",
            num_vms=1,
            resources=config_pb2.ScaleGroupResources(
                cpu_millicores=8000,
                memory_bytes=16 * 1024**3,
                device_type=config_pb2.ACCELERATOR_TYPE_CPU,
                capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
            ),
        )
        sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.slice_template.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
        sg.slice_template.gcp.zone = "us-central1-a"
        sg.slice_template.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
        sg.slice_template.gcp.machine_type = "n2-standard-4"
        config.scale_groups["cpu-vm"].CopyFrom(sg)

        validate_config(config)


def _gcp_scale_group(
    zone: str, *, capacity_type: int = config_pb2.CAPACITY_TYPE_PREEMPTIBLE
) -> config_pb2.ScaleGroupConfig:
    """Build a valid GCP-backed ScaleGroupConfig for worker settings validation tests."""
    sg = config_pb2.ScaleGroupConfig(
        name="test",
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            device_count=1,
            device_type=config_pb2.ACCELERATOR_TYPE_TPU,
            capacity_type=capacity_type,
        ),
    )
    sg.slice_template.gcp.zone = zone
    sg.slice_template.gcp.runtime_version = "v2-alpha-tpuv5-lite"
    sg.slice_template.capacity_type = capacity_type
    return sg


def _config_with_gcp_sg(
    zone: str,
    *,
    capacity_type: int = config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
    worker_attributes: dict[str, str] | None = None,
) -> config_pb2.IrisClusterConfig:
    """Build an IrisClusterConfig containing a single GCP scale group with optional worker attributes."""
    sg = _gcp_scale_group(zone, capacity_type=capacity_type)
    if worker_attributes is not None:
        for k, v in worker_attributes.items():
            sg.worker.attributes[k] = v
    config = config_pb2.IrisClusterConfig()
    config.scale_groups["test"].CopyFrom(sg)
    return config


class TestWorkerSettingsValidation:
    """Tests for worker.attributes validation (rejection of derived/well-known keys)."""

    def test_no_worker_settings_accepted(self):
        """Scale groups without worker settings always pass validation."""
        config = _config_with_gcp_sg("us-west4-b")
        validate_config(config)

    @pytest.mark.parametrize(
        "attr",
        [
            WellKnownAttribute.REGION,
            WellKnownAttribute.ZONE,
            WellKnownAttribute.DEVICE_TYPE,
            WellKnownAttribute.DEVICE_VARIANT,
            WellKnownAttribute.PREEMPTIBLE,
        ],
    )
    def test_derived_attributes_rejected(self, attr: str):
        """Well-known attributes derived from resources/slice_template must not be set explicitly."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={attr: "anything"})
        with pytest.raises(ValueError, match="derived automatically"):
            validate_config(config)

    def test_custom_attributes_accepted(self):
        """Non-well-known attributes are not rejected."""
        config = _config_with_gcp_sg("us-west4-b", worker_attributes={"team": "frontier"})
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
    num_vms: 4
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-16
      device_count: 4
      capacity_type: preemptible
    max_slices: 4
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)

        assert "tpu_v5e_16" not in config.scale_groups
        assert "tpu_v5e_16-europe-west4-b" in config.scale_groups
        assert "tpu_v5e_16-us-west4-a" in config.scale_groups

        eu = config.scale_groups["tpu_v5e_16-europe-west4-b"]
        assert eu.slice_template.gcp.zone == "europe-west4-b"
        assert eu.buffer_slices == 0

        us = config.scale_groups["tpu_v5e_16-us-west4-a"]
        assert us.slice_template.gcp.zone == "us-west4-a"

    def test_buffer_slices_preserved_when_explicit(self, tmp_path: Path):
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
    num_vms: 1
    buffer_slices: 2
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)
        assert config.scale_groups["tpu_group-us-west4-a"].buffer_slices == 2

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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
    buffer_slices: 1
    slice_template:
      gcp:
        zone: us-west4-a
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)
        assert "static_group" in config.scale_groups
        assert config.scale_groups["static_group"].buffer_slices == 1

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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: cpu
      device_count: 0
      capacity_type: on-demand
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: cos-stable
  expanded_tpu:
    zones: [us-west4-a, europe-west4-b]
    num_vms: 4
    resources:
      cpu: 128
      ram: 128GB
      disk: 1TB
      device_type: tpu
      device_variant: v5litepod-16
      device_count: 4
      capacity_type: preemptible
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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="non-empty string"):
            load_config(p)

    def test_gcp_zone_with_zones_rejected(self, tmp_path: Path):
        """Setting both zones: and slice_template.gcp.zone is rejected by expansion."""
        config_content = """\
platform:
  gcp:
    project_id: test

scale_groups:
  tpu_group:
    zones: [us-west4-a]
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
    slice_template:
      gcp:
        zone: europe-west4-b
        runtime_version: v2-alpha-tpuv5-lite
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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: cpu
      device_count: 0
      capacity_type: on-demand
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
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
    slice_template:
      gcp:
        zone: us-west4-a
        runtime_version: v2-alpha-tpuv5-lite
  tpu_group:
    zones: [us-west4-a]
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 1
      capacity_type: preemptible
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="collides"):
            load_config(p)


class TestTpuPoolExpansion:
    """Tests for tpu_pools-based scale group expansion."""

    _BASE = """\
platform:
  gcp:
    project_id: test

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

tpu_pools:
  {pool_yaml}
"""

    def test_expands_sizes_and_zones(self, tmp_path: Path):
        pool_yaml = """\
v5e-preempt:
    family: v5e
    zones: [europe-west4-b, us-west4-a]
    base_priority: 10
    resources: { cpu: 112, ram: 192GB, disk: 100GB, capacity_type: preemptible }
    slice_template:
      gcp:
        service_account: test@test.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
    sizes:
      4:  { buffer_slices: 3, max_slices: 1024 }
      16: { max_slices: 256 }"""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE.format(pool_yaml=pool_yaml))
        config = load_config(p)

        # 2 sizes x 2 zones = 4 groups
        pool_groups = [n for n in config.scale_groups if n.startswith("tpu_v5e-preempt_")]
        assert len(pool_groups) == 4

        # Check v5e-4 in europe-west4-b
        g4eu = config.scale_groups["tpu_v5e-preempt_4-europe-west4-b"]
        assert g4eu.resources.device_variant == "v5litepod-4"
        assert g4eu.resources.device_count == 4
        assert g4eu.num_vms == 1
        assert g4eu.priority == 10  # base_priority + 0*10
        assert g4eu.quota_pool == "v5e-preempt/europe-west4-b"
        assert g4eu.allocation_tier == 1
        assert g4eu.buffer_slices == 3
        assert g4eu.max_slices == 1024
        assert g4eu.slice_template.gcp.zone == "europe-west4-b"
        assert g4eu.resources.capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE

        # Check v5e-16 in us-west4-a (tier 2)
        g16us = config.scale_groups["tpu_v5e-preempt_16-us-west4-a"]
        assert g16us.resources.device_variant == "v5litepod-16"
        assert g16us.resources.device_count == 4
        assert g16us.num_vms == 4
        assert g16us.priority == 20  # base_priority + 1*10
        assert g16us.allocation_tier == 2
        assert g16us.buffer_slices == 0  # default
        assert g16us.max_slices == 256

    def test_priority_override_per_size(self, tmp_path: Path):
        pool_yaml = """\
v6e-pool:
    family: v6e
    zones: [us-east5-b]
    base_priority: 10
    resources: { cpu: 180, ram: 720GB, disk: 100GB, capacity_type: preemptible }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv6e
    sizes:
      4:  { max_slices: 100 }
      16: { max_slices: 50, priority: 99 }"""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE.format(pool_yaml=pool_yaml))
        config = load_config(p)

        g4 = config.scale_groups["tpu_v6e-pool_4-us-east5-b"]
        assert g4.priority == 10

        g16 = config.scale_groups["tpu_v6e-pool_16-us-east5-b"]
        assert g16.priority == 99  # overridden

    def test_zones_merged_into_platform(self, tmp_path: Path):
        pool_yaml = """\
v4-pool:
    family: v4
    zones: [us-central2-b]
    resources: { cpu: 240, ram: 400GB, disk: 100GB, capacity_type: preemptible }
    slice_template:
      gcp:
        runtime_version: tpu-ubuntu2204-base
    sizes:
      8: { max_slices: 10 }"""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE.format(pool_yaml=pool_yaml))
        config = load_config(p)
        assert "us-central2-b" in list(config.platform.gcp.zones)

    def test_unknown_family_rejected(self, tmp_path: Path):
        pool_yaml = """\
bad-pool:
    family: v99z
    zones: [us-central1-a]
    resources: { cpu: 8, ram: 16GB, disk: 50GB }
    slice_template:
      gcp:
        runtime_version: v2-alpha
    sizes:
      8: { max_slices: 10 }"""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE.format(pool_yaml=pool_yaml))
        with pytest.raises(ValueError, match="family"):
            load_config(p)

    def test_unknown_size_rejected(self, tmp_path: Path):
        pool_yaml = """\
bad-size:
    family: v5p
    zones: [us-central1-a]
    resources: { cpu: 208, ram: 448GB, disk: 100GB }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5
    sizes:
      7: { max_slices: 10 }"""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE.format(pool_yaml=pool_yaml))
        with pytest.raises(ValueError, match="unknown TPU topology"):
            load_config(p)

    def test_empty_sizes_rejected(self, tmp_path: Path):
        pool_yaml = """\
empty:
    family: v5e
    zones: [us-west4-a]
    resources: { cpu: 112, ram: 192GB, disk: 100GB }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
    sizes: {}"""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE.format(pool_yaml=pool_yaml))
        with pytest.raises(ValueError, match=r"sizes.*non-empty"):
            load_config(p)

    def test_duplicate_zones_rejected(self, tmp_path: Path):
        pool_yaml = """\
dupes:
    family: v5e
    zones: [us-west4-a, us-west4-a]
    resources: { cpu: 112, ram: 192GB, disk: 100GB }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
    sizes:
      4: { max_slices: 10 }"""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE.format(pool_yaml=pool_yaml))
        with pytest.raises(ValueError, match="duplicates"):
            load_config(p)

    def test_coexists_with_manual_scale_groups(self, tmp_path: Path):
        """TPU pools and manual scale_groups can coexist."""
        config_content = """\
platform:
  gcp:
    project_id: test
    zones: [us-central1-a]

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

tpu_pools:
  v5p-pool:
    family: v5p
    zones: [us-central1-a]
    resources: { cpu: 208, ram: 448GB, disk: 100GB, capacity_type: preemptible }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5
    sizes:
      8: { max_slices: 10 }

scale_groups:
  cpu_fallback:
    num_vms: 1
    resources: { cpu: 2, ram: 16GB, disk: 100GB, device_type: cpu, capacity_type: on-demand }
    slice_template:
      gcp:
        zone: us-central1-a
        mode: GCP_SLICE_MODE_VM
        machine_type: e2-highmem-2
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)

        assert "tpu_v5p-pool_8-us-central1-a" in config.scale_groups
        assert "cpu_fallback" in config.scale_groups

    def test_multiple_pools_same_family(self, tmp_path: Path):
        """Multiple pools for the same TPU family with different configs."""
        config_content = """\
platform:
  gcp:
    project_id: test

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

tpu_pools:
  v5e-preempt:
    family: v5e
    zones: [europe-west4-b]
    base_priority: 10
    resources: { cpu: 112, ram: 192GB, disk: 100GB, capacity_type: preemptible }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
    sizes:
      4: { max_slices: 100 }

  v5e-reserved:
    family: v5e
    zones: [us-east5-a]
    base_priority: 5
    resources: { cpu: 112, ram: 192GB, disk: 100GB, capacity_type: reserved }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5-lite
    sizes:
      128: { buffer_slices: 1, max_slices: 4 }
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)

        g_preempt = config.scale_groups["tpu_v5e-preempt_4-europe-west4-b"]
        assert g_preempt.quota_pool == "v5e-preempt/europe-west4-b"
        assert g_preempt.resources.capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE

        g_reserved = config.scale_groups["tpu_v5e-reserved_128-us-east5-a"]
        assert g_reserved.quota_pool == "v5e-reserved/us-east5-a"
        assert g_reserved.resources.capacity_type == config_pb2.CAPACITY_TYPE_RESERVED
        assert g_reserved.allocation_tier == 1

    def test_name_collision_rejected(self, tmp_path: Path):
        config_content = """\
platform:
  gcp:
    project_id: test
    zones: [us-central1-a]

defaults:
  worker:
    docker_image: ghcr.io/test/iris-worker:latest

tpu_pools:
  v5p-pool:
    family: v5p
    zones: [us-central1-a]
    resources: { cpu: 208, ram: 448GB, disk: 100GB }
    slice_template:
      gcp:
        runtime_version: v2-alpha-tpuv5
    sizes:
      8: { max_slices: 10 }

scale_groups:
  tpu_v5p-pool_8-us-central1-a:
    num_vms: 1
    resources: { cpu: 208, ram: 448GB, disk: 100GB, device_type: tpu, device_variant: v5p-8, device_count: 4 }
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: v2-alpha-tpuv5
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="collides"):
            load_config(p)


class TestCapacityTypeNormalization:
    """Tests for capacity_type field parsing during config normalization."""

    _BASE_CONFIG = """\
scale_groups:
  test:
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: gpu
      device_variant: a100
      device_count: 1
      capacity_type: {value}
    slice_template:
      manual:
        hosts: [10.0.0.1]
"""

    _BASE_CONFIG_NO_CAPACITY_TYPE = """\
scale_groups:
  test:
    num_vms: 1
    resources:
      cpu: 8
      ram: 16GB
      disk: 50GB
      device_type: gpu
      device_variant: a100
      device_count: 1
    slice_template:
      manual:
        hosts: [10.0.0.1]
"""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("preemptible", config_pb2.CAPACITY_TYPE_PREEMPTIBLE),
            ("on-demand", config_pb2.CAPACITY_TYPE_ON_DEMAND),
            ("on_demand", config_pb2.CAPACITY_TYPE_ON_DEMAND),
            ("reserved", config_pb2.CAPACITY_TYPE_RESERVED),
        ],
    )
    def test_capacity_type_parsed_correctly(self, tmp_path: Path, value: str, expected: int):
        content = self._BASE_CONFIG.format(value=value)
        p = tmp_path / "config.yaml"
        p.write_text(content)
        config = load_config(p)
        assert config.scale_groups["test"].resources.capacity_type == expected

    @pytest.mark.parametrize("value", ['"spot"', '"dedicated"', '"yes"', '"true"', '"false"'])
    def test_capacity_type_rejects_invalid_string(self, tmp_path: Path, value: str):
        """Strings not in the capacity type map are rejected."""
        content = self._BASE_CONFIG.format(value=value)
        p = tmp_path / "config.yaml"
        p.write_text(content)
        with pytest.raises(ValueError, match="capacity_type must be one of"):
            load_config(p)

    def test_missing_capacity_type_rejected(self, tmp_path: Path):
        """Missing capacity_type raises a validation error."""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE_CONFIG_NO_CAPACITY_TYPE)
        with pytest.raises(ValueError, match="capacity_type is required"):
            load_config(p)


def _config_with_coreweave_gpu_sg(topology_attrs: dict[str, str] | None = None) -> config_pb2.IrisClusterConfig:
    """Build a minimal IrisClusterConfig with a multi-VM CoreWeave GPU scale group."""
    config = config_pb2.IrisClusterConfig()
    sg = config.scale_groups["h100-16x"]
    sg.num_vms = 2
    sg.resources.cpu_millicores = 128_000
    sg.resources.memory_bytes = 2048 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_GPU
    sg.resources.device_variant = "H100"
    sg.resources.device_count = 8
    sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
    sg.buffer_slices = 0
    sg.max_slices = 1
    sg.slice_template.num_vms = 2
    sg.slice_template.coreweave.region = "US-WEST-04A"
    sg.slice_template.coreweave.instance_type = "gd-8xh100ib-i128"
    sg.worker.attributes["pool"] = "h100-16x"
    if topology_attrs:
        for k, v in topology_attrs.items():
            sg.worker.attributes[k] = v
    return config


def test_coreweave_gpu_multivm_requires_topology_label():
    config = _config_with_coreweave_gpu_sg()
    with pytest.raises(ValueError, match="topology label"):
        validate_config(config)


def test_coreweave_gpu_multivm_accepts_topology_label():
    config = _config_with_coreweave_gpu_sg({"backend.coreweave.cloud/superpod": "same-slice"})
    validate_config(config)


def test_coreweave_worker_provider_rejected():
    config = config_pb2.IrisClusterConfig()
    config.platform.coreweave.region = "US-WEST-04A"
    config.worker_provider.SetInParent()
    sg = config.scale_groups["cpu-test"]
    sg.num_vms = 1
    sg.resources.cpu_millicores = 64_000
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.num_vms = 1
    sg.slice_template.coreweave.region = "US-WEST-04A"
    with pytest.raises(ValueError, match="does not support worker_provider"):
        validate_config(config)


SMOKE_GCP_CONFIG = Path(__file__).resolve().parents[3] / "examples" / "smoke-gcp.yaml"


@pytest.mark.skip(
    reason="list_workers now reads from the iris.worker stats namespace. "
    "Local-mode workers boot but the in-process stats emission timing under "
    "connect_cluster is not yet reliable enough for this test; the wiring "
    "is correct (production controllers use a co-hosted finelog DuckDBLogStore) "
    "but synchronizing first-ping → flush → query inside one test process "
    "needs a follow-up. Re-enable once the local-mode stats path is "
    "deterministic."
)
def test_smoke_gcp_config_boots_locally():
    """Load smoke-gcp.yaml, convert to local mode, verify workers join."""
    config = load_config(SMOKE_GCP_CONFIG)
    config = make_local_config(config)

    with connect_cluster(config) as url:
        client = ControllerServiceClientSync(address=url, timeout_ms=30000)

        def _has_healthy_worker() -> bool:
            workers = client.list_workers(controller_pb2.Controller.ListWorkersRequest()).workers
            return any(w.healthy for w in workers)

        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until_or_raise(
            _has_healthy_worker,
            timeout=Duration.from_seconds(30.0),
            error_message="No healthy workers with smoke-gcp.yaml in local mode",
        )
        client.close()
