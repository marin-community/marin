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
    BackendConfig,
    ControllerVmConfig,
    CoreweavePlatformConfig,
    CoreweaveSliceConfig,
    DefaultsConfig,
    GcpControllerConfig,
    GcpPlatformConfig,
    GcpSliceConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    LocalSliceConfig,
    ManualSliceConfig,
    PlatformConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    SshConfig,
    WorkerConfig,
    WorkerProviderConfig,
    WorkerSettings,
    backend_attribute_sets,
    build_ssh_command_config,
    config_to_dict,
    load_config,
    make_local_config,
    resolve_backends,
    validate_config,
)
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.autoscaler.factory import create_autoscaler
from iris.cluster.lifecycle import connect_cluster
from iris.cluster.platforms.factory import create_provider_bundle
from iris.cluster.platforms.gcp.service import KNOWN_GCP_ZONES
from iris.cluster.types import DEFAULT_BACKEND_ID, LOCAL_CLUSTER, AcceleratorType, CapacityType, GcpSliceMode
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from rigging.timing import Duration, ExponentialBackoff


class TestConfigRoundTrip:
    """Tests for config serialization/deserialization round-trips."""

    def test_tpu_provider_survives_round_trip(self, tmp_path: Path):
        """TPU config survives proto→dict→yaml→dict→proto round-trip."""
        config_content = """\
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        # Load config from YAML
        original_config = load_config(config_path)

        # Verify accelerator type
        assert original_config.scale_groups["tpu_v5e_8"].resources.device_type == AcceleratorType.TPU

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify accelerator type is still TPU after round-trip
        assert loaded_config.scale_groups["tpu_v5e_8"].resources.device_type == AcceleratorType.TPU

    def test_manual_provider_survives_round_trip(self, tmp_path: Path):
        """Manual config survives proto→dict→yaml→dict→proto round-trip."""
        config_content = """\
name: test-cluster

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
        assert original_config.scale_groups["manual_hosts"].slice_template is not None
        assert original_config.scale_groups["manual_hosts"].slice_template.platform_kind() == "manual"
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
        assert loaded_config.scale_groups["manual_hosts"].slice_template is not None
        assert loaded_config.scale_groups["manual_hosts"].slice_template.platform_kind() == "manual"
        assert list(loaded_config.scale_groups["manual_hosts"].slice_template.manual.hosts) == [
            "10.0.0.1",
            "10.0.0.2",
        ]

    def test_multiple_scale_groups_preserve_accelerator_types(self, tmp_path: Path):
        """Config with multiple TPU scale groups preserves accelerator types."""
        config_content = """\
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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

        assert loaded_config.scale_groups["tpu_group_a"].resources.device_type == AcceleratorType.TPU
        assert loaded_config.scale_groups["tpu_group_b"].resources.device_type == AcceleratorType.TPU

    def test_example_eu_west4_config_round_trips(self, tmp_path: Path):
        """Real example config from config/marin.yaml round-trips correctly."""
        iris_root = Path(__file__).parent.parent.parent.parent
        config_path = iris_root / "config" / "marin.yaml"
        if not config_path.exists():
            pytest.skip("Example config not found")

        original_config = load_config(config_path)

        # After expansion, zone-specific groups should exist
        assert "tpu_v5e-preemptible_16-europe-west4-b" in original_config.scale_groups
        assert (
            original_config.scale_groups["tpu_v5e-preemptible_16-europe-west4-b"].resources.device_type
            == AcceleratorType.TPU
        )

        # Round-trip via dict and YAML
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        assert (
            loaded_config.scale_groups["tpu_v5e-preemptible_16-europe-west4-b"].resources.device_type
            == AcceleratorType.TPU
        )

    @pytest.mark.parametrize(
        "accelerator_type,expected_enum",
        [
            ("tpu", AcceleratorType.TPU),
            ("cpu", AcceleratorType.CPU),
            ("gpu", AcceleratorType.GPU),
        ],
    )
    def test_lowercase_accelerator_types_work(self, tmp_path: Path, accelerator_type: str, expected_enum):
        """Config accepts lowercase accelerator types in resources.device_type."""
        config_content = f"""\
name: test-cluster

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

    def test_proto_name_accelerator_types_rejected(self, tmp_path: Path):
        """device_type only accepts the friendly lowercase spelling (cpu/gpu/tpu)."""
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match=r"device_type[\s\S]*'cpu', 'gpu' or 'tpu'"):
            load_config(config_path)


class TestCreateAutoscalerFromConfig:
    """Tests for create_autoscaler factory function."""

    def test_creates_autoscaler_with_tpu_provider(self, tmp_path: Path):
        """create_autoscaler works with TPU config."""
        config_content = """\
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
        zone: us-central1-a
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        bundle = create_provider_bundle(
            platform_config=config.platform,
            worker_port=config.defaults.worker.port,
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
name: test-cluster

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
            worker_port=config.defaults.worker.port,
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
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
            worker_port=loaded_config.defaults.worker.port,
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
        """build_ssh_command_config returns cluster defaults when no group override."""

        config = IrisClusterConfig()
        config.defaults.ssh = SshConfig(
            user="ubuntu",
            key_file="~/.ssh/cluster_key",
            impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
            connect_timeout=Duration.from_seconds(60),
        )

        ssh_config = build_ssh_command_config(config)

        assert ssh_config.user == "ubuntu"
        assert ssh_config.key_file == "~/.ssh/cluster_key"
        assert ssh_config.port == 22  # DEFAULT_SSH_PORT
        assert ssh_config.impersonate_service_account == "iris-controller@test-project.iam.gserviceaccount.com"
        assert ssh_config.connect_timeout.to_ms() == 60_000

    def test_applies_per_group_ssh_overrides(self):
        """build_ssh_command_config applies per-group SSH overrides for manual slice template."""
        config = IrisClusterConfig()
        config.defaults.ssh.user = "ubuntu"
        config.defaults.ssh.key_file = "~/.ssh/cluster_key"

        config.scale_groups["manual_group"] = ScaleGroupConfig(
            name="manual_group",
            slice_template=SliceConfig(
                manual=ManualSliceConfig(
                    hosts=["10.0.0.1"],
                    ssh_user="admin",
                    ssh_key_file="~/.ssh/group_key",
                )
            ),
        )

        ssh_config = build_ssh_command_config(config, group_name="manual_group")

        assert ssh_config.user == "admin"
        assert ssh_config.key_file == "~/.ssh/group_key"
        assert ssh_config.port == 22

    def test_uses_defaults_when_cluster_ssh_config_empty(self):
        """build_ssh_command_config uses built-in defaults when cluster config empty."""

        config = IrisClusterConfig()

        ssh_config = build_ssh_command_config(config)

        assert ssh_config.user == "root"
        assert ssh_config.key_file == ""
        assert ssh_config.port == 22
        assert ssh_config.impersonate_service_account == ""
        assert ssh_config.connect_timeout.to_ms() == 30_000

    def test_validate_config_requires_gcp_service_accounts(self):
        config = IrisClusterConfig(
            name="test-cluster",
            platform=PlatformConfig(gcp=GcpPlatformConfig(project_id="test-project")),
            controller=ControllerVmConfig(gcp=GcpControllerConfig(zone="us-central1-a")),
        )
        config.defaults.worker.docker_image = "ghcr.io/marin-community/iris-worker:latest"

        config.scale_groups["tpu"] = ScaleGroupConfig(
            name="tpu",
            num_vms=1,
            resources=ScaleGroupResources(
                device_type=AcceleratorType.TPU,
                device_variant="v5litepod-4",
                capacity_type=CapacityType.PREEMPTIBLE,
            ),
            slice_template=SliceConfig(
                gcp=GcpSliceConfig(zone="us-central1-a", runtime_version="tpu-ubuntu2204-base"),
            ),
        )

        with pytest.raises(ValueError, match=r"controller\.gcp\.service_account"):
            validate_config(config)


class TestLocalConfigTransformation:
    """Tests for make_local_config transformation."""

    def test_make_local_config_transforms_gcp_to_local(self, tmp_path: Path):
        """make_local_config transforms GCP config to local mode."""

        config_content = """\
name: test-cluster

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
    service_account: iris-controller@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        zone: us-central1-a
        runtime_version: tpu-ubuntu2204-base
"""
        config_path = tmp_path / "gcp_config.yaml"
        config_path.write_text(config_content)

        # Load and transform
        original_config = load_config(config_path)
        local_config = make_local_config(original_config)

        # Verify platform transformed to local
        assert local_config.platform.platform_kind() == "local"

        # Verify controller transformed to local
        assert local_config.controller.controller_kind() == "local"
        assert local_config.controller.local.port == 0  # auto-assign

        # Verify fast local-dev timings override any production timings.
        assert local_config.defaults.autoscaler.evaluation_interval.to_ms() == 500
        assert local_config.defaults.autoscaler.scale_up_delay.to_ms() == 1000
        assert local_config.defaults.autoscaler.scale_down_delay.to_ms() == 1000

    def test_make_local_config_preserves_scale_group_details(self, tmp_path: Path):
        """make_local_config preserves accelerator type and other scale group settings."""

        config_content = """\
name: test-cluster

platform:
  gcp:
    project_id: test-project

defaults:
  worker:
    docker_image: gcr.io/test/worker:latest

controller:
  gcp:
    service_account: iris-controller@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        zone: us-central1-a
        runtime_version: tpu-ubuntu2204-base
"""
        config_path = tmp_path / "multi_group.yaml"
        config_path.write_text(config_content)

        original_config = load_config(config_path)
        local_config = make_local_config(original_config)

        # Verify other fields preserved
        cpu_group = local_config.scale_groups["cpu_group"]
        assert cpu_group.resources.device_type == AcceleratorType.CPU
        assert cpu_group.buffer_slices == 2
        assert cpu_group.max_slices == 5
        assert cpu_group.priority == 50

        tpu_group = local_config.scale_groups["tpu_group"]
        assert tpu_group.resources.device_type == AcceleratorType.TPU
        assert tpu_group.resources.device_variant == "v5litepod-16"
        assert tpu_group.buffer_slices == 1
        assert tpu_group.max_slices == 3
        assert tpu_group.priority == 100

    def test_example_configs_load_and_transform(self):
        """Example configs in config/ directory load and transform to local correctly."""

        iris_root = Path(__file__).parent.parent.parent.parent
        example_configs = [
            iris_root / "config" / "marin.yaml",
            iris_root / "config" / "marin-dev.yaml",
            iris_root / "config" / "examples" / "coreweave.yaml",
            iris_root / "config" / "ci-coreweave.yaml",
            iris_root / "config" / "ci-test.yaml",
        ]

        for config_path in example_configs:
            if not config_path.exists():
                pytest.skip(f"Example config not found: {config_path}")

            # Load the config
            config = load_config(config_path)
            assert config.platform.platform_kind() in ["gcp", "manual", "coreweave"]
            assert config.defaults.autoscaler.evaluation_interval.to_ms() > 0

            # Transform to local
            local_config = make_local_config(config)
            assert local_config.platform.platform_kind() == "local"
            assert local_config.controller.controller_kind() == "local"
            # Verify fast timings applied
            assert local_config.defaults.autoscaler.evaluation_interval.to_ms() == 500
            assert local_config.defaults.autoscaler.scale_up_delay.to_ms() == 1000

    def test_example_config_zones_in_known_gcp_zones(self):
        """All GCP zones used in example configs must be in KNOWN_GCP_ZONES."""

        iris_root = Path(__file__).parent.parent.parent.parent
        for config_path in [iris_root / "config" / "marin.yaml", iris_root / "config" / "marin-dev.yaml"]:
            if not config_path.exists():
                pytest.skip(f"Example config not found: {config_path}")
            config = load_config(config_path)
            for name, sg in config.scale_groups.items():
                template = sg.slice_template
                if template.platform_kind() == "gcp" and template.gcp.zone:
                    assert (
                        template.gcp.zone in KNOWN_GCP_ZONES
                    ), f"Scale group '{name}': zone '{template.gcp.zone}' not in KNOWN_GCP_ZONES"


def _valid_scale_group() -> ScaleGroupConfig:
    """Create a valid ScaleGroupConfig for use in validation tests."""
    return ScaleGroupConfig(
        name="test",
        num_vms=1,
        resources=ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            device_type=AcceleratorType.CPU,
            capacity_type=CapacityType.ON_DEMAND,
        ),
        slice_template=SliceConfig(num_vms=1, local=LocalSliceConfig()),
    )


def _config_with(**overrides) -> IrisClusterConfig:
    """Build an IrisClusterConfig with a single scale group, overriding fields."""
    sg = _valid_scale_group()
    for key, value in overrides.items():
        setattr(sg, key, value)
    return IrisClusterConfig(name="test-cluster", scale_groups={"test": sg})


class TestOneofExclusivity:
    """Former protobuf ``oneof`` groups reject more than one selected arm."""

    def test_construct_two_platform_arms_rejected(self):
        with pytest.raises(ValueError, match="at most one of"):
            PlatformConfig(gcp=GcpPlatformConfig(), coreweave=CoreweavePlatformConfig())

    def test_construct_two_slice_arms_rejected(self):
        with pytest.raises(ValueError, match="at most one of"):
            SliceConfig(manual=ManualSliceConfig(), local=LocalSliceConfig())

    def test_yaml_two_platform_arms_rejected(self, tmp_path: Path):
        config_path = tmp_path / "cluster.yaml"
        config_path.write_text(
            yaml.safe_dump({"platform": {"gcp": {"project_id": "p"}, "manual": {}}}),
        )
        with pytest.raises(ValueError, match="at most one of"):
            load_config(config_path)

    def test_zero_arms_allowed(self):
        # A platform with no selected arm constructs; downstream validation owns
        # the "a platform is required" rule.
        assert PlatformConfig().platform_kind() is None

    def test_exclude_none_dump_round_trips_single_arm(self):
        # config_to_dict serializes with exclude_none; the unselected arms must
        # not reappear on re-validation and trip the at-most-one check.
        dumped = PlatformConfig(gcp=GcpPlatformConfig(project_id="p")).model_dump(mode="json", exclude_none=True)
        assert PlatformConfig.model_validate(dumped).platform_kind() == "gcp"


class TestConfigValidation:
    """Tests for validate_config: the consolidated entry point for config validation."""

    def test_valid_config_accepted(self):
        validate_config(_config_with())

    def test_rejects_missing_cluster_name(self):
        with pytest.raises(ValueError, match="cluster name is required"):
            validate_config(IrisClusterConfig())

    def test_rejects_the_federation_sentinel_as_a_cluster_name(self):
        # 'local' means "this controller" to federation; a real cluster claiming it
        # would collide with the sentinel in the cluster-id namespace.
        with pytest.raises(ValueError, match="reserved as the federation"):
            validate_config(IrisClusterConfig(name=LOCAL_CLUSTER))

    def test_rejects_missing_resources(self):
        config = IrisClusterConfig(name="test-cluster", scale_groups={"test": ScaleGroupConfig(name="test", num_vms=1)})
        with pytest.raises(ValueError, match="must set resources"):
            validate_config(config)

    def test_rejects_missing_num_vms(self):
        config = IrisClusterConfig(
            name="test-cluster",
            scale_groups={
                "test": ScaleGroupConfig(
                    name="test",
                    resources=ScaleGroupResources(
                        cpu_millicores=8000,
                        memory_bytes=16 * 1024**3,
                        device_type=AcceleratorType.CPU,
                    ),
                )
            },
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
                    resources=ScaleGroupResources(
                        cpu_millicores=8000,
                        memory_bytes=16 * 1024**3,
                        device_type=None,
                    )
                )
            )

    def test_rejects_negative_cpu(self):
        with pytest.raises(ValueError, match="invalid cpu_millicores"):
            validate_config(
                _config_with(
                    resources=ScaleGroupResources(
                        cpu_millicores=-1000,
                        memory_bytes=16 * 1024**3,
                        device_type=AcceleratorType.CPU,
                    )
                )
            )

    def test_rejects_gcp_zone_not_in_platform_zones(self):
        """Validation fails when scale group zone is not in platform.gcp.zones."""
        config = IrisClusterConfig(
            name="test-cluster",
            platform=PlatformConfig(gcp=GcpPlatformConfig(project_id="test", zones=["zone-a"])),
            scale_groups={
                "tpu": ScaleGroupConfig(
                    name="tpu",
                    num_vms=8,
                    resources=ScaleGroupResources(
                        cpu_millicores=8000,
                        memory_bytes=16 * 1024**3,
                        device_count=4,
                        device_type=AcceleratorType.TPU,
                        capacity_type=CapacityType.PREEMPTIBLE,
                    ),
                    slice_template=SliceConfig(
                        gcp=GcpSliceConfig(zone="zone-b", runtime_version="v2-alpha-tpuv5-lite"),
                    ),
                )
            },
        )

        with pytest.raises(ValueError, match=r"not in platform\.gcp\.zones"):
            validate_config(config)

    def test_accepts_gcp_zone_in_platform_zones(self):
        """Validation passes when scale group zone is in platform.gcp.zones."""
        config = IrisClusterConfig(
            name="test-cluster",
            platform=PlatformConfig(gcp=GcpPlatformConfig(project_id="test", zones=["zone-a"])),
            defaults=DefaultsConfig(worker=WorkerConfig(docker_image="ghcr.io/marin/iris-worker:latest")),
            scale_groups={
                "tpu": ScaleGroupConfig(
                    name="tpu",
                    num_vms=8,
                    resources=ScaleGroupResources(
                        cpu_millicores=8000,
                        memory_bytes=16 * 1024**3,
                        device_count=4,
                        device_type=AcceleratorType.TPU,
                        capacity_type=CapacityType.PREEMPTIBLE,
                    ),
                    slice_template=SliceConfig(
                        gcp=GcpSliceConfig(
                            zone="zone-a",
                            runtime_version="v2-alpha-tpuv5-lite",
                            service_account="iris-worker@test-project.iam.gserviceaccount.com",
                        ),
                    ),
                )
            },
        )

        validate_config(config)  # Should not raise

    @pytest.mark.parametrize(
        "num_vms,device_type,device_count,capacity_type,error_match",
        [
            (
                1,
                AcceleratorType.CPU,
                0,
                CapacityType.PREEMPTIBLE,
                "only support capacity_type on-demand",
            ),
            (2, AcceleratorType.CPU, 0, CapacityType.ON_DEMAND, "require num_vms=1"),
            (1, AcceleratorType.GPU, 1, CapacityType.ON_DEMAND, "require device_type=cpu"),
        ],
        ids=["non_on_demand", "multi_vm", "non_cpu"],
    )
    def test_rejects_invalid_gcp_vm_mode(self, num_vms, device_type, device_count, capacity_type, error_match):
        config = IrisClusterConfig(
            name="test-cluster",
            scale_groups={
                "test-vm": ScaleGroupConfig(
                    name="test-vm",
                    num_vms=num_vms,
                    resources=ScaleGroupResources(
                        cpu_millicores=8000,
                        memory_bytes=16 * 1024**3,
                        device_type=device_type,
                        device_count=device_count,
                        capacity_type=capacity_type,
                    ),
                    slice_template=SliceConfig(
                        gcp=GcpSliceConfig(
                            zone="us-central1-a",
                            mode=GcpSliceMode.VM,
                            machine_type="n2-standard-4",
                        ),
                    ),
                )
            },
        )

        with pytest.raises(ValueError, match=error_match):
            validate_config(config)

    def test_accepts_gcp_vm_mode_cpu_single_vm_on_demand(self):
        config = IrisClusterConfig(
            name="test-cluster",
            scale_groups={
                "cpu-vm": ScaleGroupConfig(
                    name="cpu-vm",
                    num_vms=1,
                    resources=ScaleGroupResources(
                        cpu_millicores=8000,
                        memory_bytes=16 * 1024**3,
                        device_type=AcceleratorType.CPU,
                        capacity_type=CapacityType.ON_DEMAND,
                    ),
                    slice_template=SliceConfig(
                        gcp=GcpSliceConfig(
                            zone="us-central1-a",
                            mode=GcpSliceMode.VM,
                            machine_type="n2-standard-4",
                            service_account="iris-worker@test-project.iam.gserviceaccount.com",
                        ),
                    ),
                )
            },
        )

        validate_config(config)


def _gcp_scale_group(zone: str, *, capacity_type: CapacityType = CapacityType.PREEMPTIBLE) -> ScaleGroupConfig:
    """Build a valid GCP-backed ScaleGroupConfig for worker settings validation tests."""
    return ScaleGroupConfig(
        name="test",
        num_vms=1,
        resources=ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            device_count=1,
            device_type=AcceleratorType.TPU,
            capacity_type=capacity_type,
        ),
        slice_template=SliceConfig(
            gcp=GcpSliceConfig(
                zone=zone,
                runtime_version="v2-alpha-tpuv5-lite",
                service_account="iris-worker@test-project.iam.gserviceaccount.com",
            ),
        ),
    )


def _config_with_gcp_sg(
    zone: str,
    *,
    capacity_type: CapacityType = CapacityType.PREEMPTIBLE,
    worker_attributes: dict[str, str] | None = None,
) -> IrisClusterConfig:
    """Build an IrisClusterConfig containing a single GCP scale group with optional worker attributes."""
    sg = _gcp_scale_group(zone, capacity_type=capacity_type)
    if worker_attributes is not None:
        sg.worker = WorkerSettings(attributes=dict(worker_attributes))
    return IrisClusterConfig(name="test-cluster", scale_groups={"test": sg})


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
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)
        assert config.scale_groups["tpu_group-us-west4-a"].buffer_slices == 2

    def test_groups_without_zones_unchanged(self, tmp_path: Path):
        config_content = """\
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="non-empty"):
            load_config(p)

    def test_mixed_expanded_and_static_groups(self, tmp_path: Path):
        """Expanded and non-expanded groups coexist."""
        config_content = """\
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        with pytest.raises(ValueError, match="collides"):
            load_config(p)


class TestTpuPoolExpansion:
    """Tests for tpu_pools-based scale group expansion."""

    _BASE = """\
name: test-cluster

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
        assert g4eu.resources.capacity_type == CapacityType.PREEMPTIBLE

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5
    sizes:
      8: { max_slices: 10 }

scale_groups:
  cpu_fallback:
    num_vms: 1
    resources: { cpu: 2, ram: 16GB, disk: 100GB, device_type: cpu, capacity_type: on-demand }
    slice_template:
      gcp:
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        zone: us-central1-a
        mode: vm
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
name: test-cluster

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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
    sizes:
      128: { buffer_slices: 1, max_slices: 4 }
"""
        p = tmp_path / "config.yaml"
        p.write_text(config_content)
        config = load_config(p)

        g_preempt = config.scale_groups["tpu_v5e-preempt_4-europe-west4-b"]
        assert g_preempt.quota_pool == "v5e-preempt/europe-west4-b"
        assert g_preempt.resources.capacity_type == CapacityType.PREEMPTIBLE

        g_reserved = config.scale_groups["tpu_v5e-reserved_128-us-east5-a"]
        assert g_reserved.quota_pool == "v5e-reserved/us-east5-a"
        assert g_reserved.resources.capacity_type == CapacityType.RESERVED
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
        service_account: iris-worker@test-project.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5
    sizes:
      8: { max_slices: 10 }

scale_groups:
  tpu_v5p-pool_8-us-central1-a:
    num_vms: 1
    resources: { cpu: 208, ram: 448GB, disk: 100GB, device_type: tpu, device_variant: v5p-8, device_count: 4 }
    slice_template:
      gcp:
        service_account: iris-worker@test-project.iam.gserviceaccount.com
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
name: test-cluster

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
name: test-cluster

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
            ("preemptible", CapacityType.PREEMPTIBLE),
            ("on-demand", CapacityType.ON_DEMAND),
            ("on_demand", CapacityType.ON_DEMAND),
            ("reserved", CapacityType.RESERVED),
        ],
    )
    def test_capacity_type_parsed_correctly(self, tmp_path: Path, value: str, expected: CapacityType):
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
        with pytest.raises(ValueError, match=r"capacity_type[\s\S]*'preemptible', 'on_demand' or 'reserved'"):
            load_config(p)

    def test_missing_capacity_type_rejected(self, tmp_path: Path):
        """Missing capacity_type raises a validation error."""
        p = tmp_path / "config.yaml"
        p.write_text(self._BASE_CONFIG_NO_CAPACITY_TYPE)
        with pytest.raises(ValueError, match="capacity_type is required"):
            load_config(p)


def _config_with_coreweave_gpu_sg(topology_attrs: dict[str, str] | None = None) -> IrisClusterConfig:
    """Build a minimal IrisClusterConfig with a multi-VM CoreWeave GPU scale group."""
    attributes = {"pool": "h100-16x"}
    if topology_attrs:
        attributes.update(topology_attrs)
    sg = ScaleGroupConfig(
        name="h100-16x",
        num_vms=2,
        buffer_slices=0,
        max_slices=1,
        resources=ScaleGroupResources(
            cpu_millicores=128_000,
            memory_bytes=2048 * 1024**3,
            device_type=AcceleratorType.GPU,
            device_variant="H100",
            device_count=8,
            capacity_type=CapacityType.ON_DEMAND,
        ),
        slice_template=SliceConfig(
            num_vms=2,
            coreweave=CoreweaveSliceConfig(region="US-WEST-04A", instance_type="gd-8xh100ib-i128"),
        ),
        worker=WorkerSettings(attributes=attributes),
    )
    return IrisClusterConfig(name="test-cluster", scale_groups={"h100-16x": sg})


def test_coreweave_gpu_multivm_requires_topology_label():
    config = _config_with_coreweave_gpu_sg()
    with pytest.raises(ValueError, match="topology label"):
        validate_config(config)


def test_coreweave_gpu_multivm_accepts_topology_label():
    config = _config_with_coreweave_gpu_sg({"backend.coreweave.cloud/superpod": "same-slice"})
    validate_config(config)


def test_coreweave_worker_provider_rejected():
    config = IrisClusterConfig(
        name="test-cluster",
        platform=PlatformConfig(coreweave=CoreweavePlatformConfig(region="US-WEST-04A")),
        worker_provider=WorkerProviderConfig(),
        scale_groups={
            "cpu-test": ScaleGroupConfig(
                name="cpu-test",
                num_vms=1,
                resources=ScaleGroupResources(cpu_millicores=64_000, device_type=AcceleratorType.CPU),
                slice_template=SliceConfig(num_vms=1, coreweave=CoreweaveSliceConfig(region="US-WEST-04A")),
            )
        },
    )
    with pytest.raises(ValueError, match="does not support worker_provider"):
        validate_config(config)


SMOKE_GCP_CONFIG = Path(__file__).resolve().parents[3] / "config" / "ci-gcp-smoke.yaml"


@pytest.mark.timeout(15)
def test_smoke_gcp_config_boots_locally():
    """Load ci-gcp-smoke.yaml, convert to local mode, verify workers join."""
    config = load_config(SMOKE_GCP_CONFIG)
    config = make_local_config(config)

    with connect_cluster(config) as url:
        client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        # The smoke config has buffer_slices=1 for v5e-smoke/16 across 2 zones,
        # each with num_vms=4 → 8 workers total.  We only need one healthy
        # worker to confirm the config boots.

        def _has_healthy_worker() -> bool:
            workers = client.list_workers(controller_pb2.Controller.ListWorkersRequest()).workers
            return any(w.healthy for w in workers)

        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until_or_raise(
            _has_healthy_worker,
            timeout=Duration.from_seconds(15.0),
            error_message="No healthy workers with ci-gcp-smoke.yaml in local mode",
        )
        client.close()


def _worker_daemon_backend(**overrides) -> BackendConfig:
    """A minimal valid worker_daemon backend (worker_provider present, in_process)."""
    fields = {"kind": "worker_daemon", "worker_provider": WorkerProviderConfig()}
    fields.update(overrides)
    return BackendConfig(**fields)


def _accel_scale_group(device_type: AcceleratorType, device_variant: str = "") -> ScaleGroupConfig:
    """A scale group carrying the device fields backend attribute derivation reads."""
    return ScaleGroupConfig(
        name="accel",
        num_vms=1,
        resources=ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            device_type=device_type,
            device_variant=device_variant,
            capacity_type=CapacityType.ON_DEMAND,
        ),
    )


class TestBackendsConfig:
    """The explicit ``backends:`` map and the implicit single-backend synthesis."""

    def test_explicit_backends_map_validates_and_resolves(self):
        config = IrisClusterConfig(
            name="test-cluster",
            backends={"cpu": _worker_daemon_backend(attributes={"device-type": "cpu"})},
        )
        validate_config(config)

        resolved = resolve_backends(config)
        assert set(resolved) == {"cpu"}
        # Defaults fill in: in_process transport.
        assert resolved["cpu"].transport == "in_process"

    def test_mixing_backends_with_top_level_scale_groups_rejected(self):
        config = IrisClusterConfig(
            name="test-cluster",
            backends={"cpu": _worker_daemon_backend()},
            scale_groups={"test": _valid_scale_group()},
        )
        with pytest.raises(ValueError, match="cannot be combined with top-level"):
            validate_config(config)

    def test_multiple_in_process_backends_validate(self):
        # The controller routes tasks to N in-process backends by backend_id and
        # drives them from its single control thread, so more than one in_process
        # backend is allowed.
        config = IrisClusterConfig(
            name="test-cluster",
            backends={
                "cpu": _worker_daemon_backend(attributes={"device-type": "cpu"}),
                "tpu": _worker_daemon_backend(attributes={"device-type": "tpu"}),
            },
        )
        validate_config(config)

        resolved = resolve_backends(config)
        assert set(resolved) == {"cpu", "tpu"}

    def test_remote_transport_rejected(self):
        config = IrisClusterConfig(
            name="test-cluster",
            backends={"cpu": _worker_daemon_backend(transport="remote")},
        )
        with pytest.raises(ValueError, match="remote transport lands in a later PR"):
            validate_config(config)

    def test_k8s_backend_requires_kubernetes_provider(self):
        config = IrisClusterConfig(name="test-cluster", backends={"k8s": BackendConfig(kind="k8s")})
        with pytest.raises(ValueError, match="kind 'k8s' requires kubernetes_provider"):
            validate_config(config)

    def test_backend_scale_group_names_injected(self, tmp_path: Path):
        # A backend's scale groups must get their map key stamped onto `name`,
        # exactly like top-level scale groups. Otherwise the worker registers
        # under an empty scale group, which maps to DEFAULT_BACKEND_ID instead
        # of the backend that owns it, and the backend never sees its workers.
        config_path = tmp_path / "cluster.yaml"
        config_path.write_text(
            "name: c\n"
            "backends:\n"
            "  tpu:\n"
            "    kind: worker_daemon\n"
            "    worker_provider: {}\n"
            "    scale_groups:\n"
            "      sg-tpu:\n"
            "        max_slices: 1\n"
        )
        config = load_config(config_path)
        assert config.backends["tpu"].scale_groups["sg-tpu"].name == "sg-tpu"

    def test_single_cluster_synthesizes_one_default_backend(self):
        config = _config_with()  # no backends: — implicit single-backend form
        validate_config(config)

        resolved = resolve_backends(config)
        assert list(resolved) == [DEFAULT_BACKEND_ID]
        synthesized = resolved[DEFAULT_BACKEND_ID]
        assert synthesized.kind == "worker_daemon"
        assert set(synthesized.scale_groups) == set(config.scale_groups)

    def test_kubernetes_provider_synthesizes_k8s_backend(self):
        config = IrisClusterConfig(
            kubernetes_provider=KubernetesProviderConfig(controller_address="http://controller:10000"),
        )
        resolved = resolve_backends(config)
        assert resolved[DEFAULT_BACKEND_ID].kind == "k8s"

    def test_attribute_values_comma_split_into_sets(self):
        backend = _worker_daemon_backend(attributes={"device-variant": "v5e-4, v5p-8 ,", "device-type": "tpu"})
        assert backend_attribute_sets(backend) == {
            "device-variant": {"v5e-4", "v5p-8"},
            "device-type": {"tpu"},
        }

    def test_device_attrs_derived_from_scale_group_resources(self):
        # A GPU scale group with no explicit backend attributes advertises the
        # device-type/device-variant its resources declare, lowercased ('H100' -> 'h100')
        # so the value equals the constraint literal a GPU job matches with.
        backend = _worker_daemon_backend(scale_groups={"h100-8x": _accel_scale_group(AcceleratorType.GPU, "H100")})
        assert backend_attribute_sets(backend) == {"device-type": {"gpu"}, "device-variant": {"h100"}}

    def test_cpu_scale_group_advertises_no_device_attrs(self):
        # A CPU-only backend derives nothing, staying a routing catch-all as before.
        backend = _worker_daemon_backend(scale_groups={"cpu": _accel_scale_group(AcceleratorType.CPU)})
        assert backend_attribute_sets(backend) == {}

    def test_multi_scale_group_backend_advertises_variant_union(self):
        # Several accelerator groups union their variants; a CPU group adds nothing.
        backend = _worker_daemon_backend(
            scale_groups={
                "h100": _accel_scale_group(AcceleratorType.GPU, "H100"),
                "a100": _accel_scale_group(AcceleratorType.GPU, "A100"),
                "cpu": _accel_scale_group(AcceleratorType.CPU),
            }
        )
        assert backend_attribute_sets(backend) == {"device-type": {"gpu"}, "device-variant": {"h100", "a100"}}

    def test_derived_device_attrs_union_with_explicit_attributes(self):
        # Explicit non-device attributes are preserved; an explicit device-variant
        # unions with the derived one rather than being overwritten.
        backend = _worker_daemon_backend(
            attributes={"pool": "h100-8x", "device-variant": "a100"},
            scale_groups={"h100": _accel_scale_group(AcceleratorType.GPU, "H100")},
        )
        assert backend_attribute_sets(backend) == {
            "pool": {"h100-8x"},
            "device-type": {"gpu"},
            "device-variant": {"a100", "h100"},
        }

    def test_auto_variant_derives_device_type_only(self):
        # A blank or 'auto' variant yields device-type but no device-variant,
        # matching constraints_from_resources which emits no variant constraint.
        backend = _worker_daemon_backend(scale_groups={"tpu": _accel_scale_group(AcceleratorType.TPU, "auto")})
        assert backend_attribute_sets(backend) == {"device-type": {"tpu"}}

    def test_coreweave_implicit_config_advertises_gpu_attrs(self):
        # The CoreWeave configs use the implicit single-backend shape (no backends:).
        # resolve_backends synthesizes one backend whose GPU scale group must
        # advertise device-type/device-variant, else a GPU job can neither route to
        # it locally nor federate to it as a peer.
        iris_root = Path(__file__).parent.parent.parent.parent
        for rel in ("config/examples/coreweave.yaml", "config/cw-us-east-02a.yaml"):
            config_path = iris_root / rel
            if not config_path.exists():
                pytest.skip(f"Config not found: {rel}")
            config = load_config(config_path)
            resolved = resolve_backends(config)
            assert list(resolved) == [DEFAULT_BACKEND_ID]
            assert backend_attribute_sets(resolved[DEFAULT_BACKEND_ID]) == {
                "device-type": {"gpu"},
                "device-variant": {"h100"},
            }
