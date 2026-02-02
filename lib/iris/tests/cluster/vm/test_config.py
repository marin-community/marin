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
YAML, ensuring that proto oneofs (like provider type) are preserved correctly.
"""

from pathlib import Path

import pytest
import yaml
from google.protobuf.json_format import ParseDict

from iris.cluster.vm.config import (
    config_to_dict,
    create_autoscaler_from_config,
    get_ssh_config,
    load_config,
)
from iris.rpc import config_pb2


class TestConfigRoundTrip:
    """Tests for config serialization/deserialization round-trips."""

    def test_tpu_provider_survives_round_trip(self, tmp_path: Path):
        """TPU provider config survives proto→dict→yaml→dict→proto round-trip."""
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
    provider:
      tpu:
        project_id: my-project
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

        # Verify provider is TPU before round-trip
        assert original_config.scale_groups["tpu_v5e_8"].HasField("provider")
        provider = original_config.scale_groups["tpu_v5e_8"].provider
        assert provider.WhichOneof("provider") == "tpu"
        assert provider.tpu.project_id == "my-project"

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify provider is still TPU after round-trip
        assert loaded_config.scale_groups["tpu_v5e_8"].HasField("provider")
        provider_after = loaded_config.scale_groups["tpu_v5e_8"].provider
        assert (
            provider_after.WhichOneof("provider") == "tpu"
        ), f"Expected provider to be 'tpu' after round-trip, got '{provider_after.WhichOneof('provider')}'"
        assert provider_after.tpu.project_id == "my-project"

    def test_manual_provider_survives_round_trip(self, tmp_path: Path):
        """Manual provider config survives proto→dict→yaml→dict→proto round-trip."""
        config_content = """\
provider_type: manual

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

ssh:
  user: ubuntu
  key_file: ~/.ssh/id_rsa

scale_groups:
  manual_hosts:
    provider:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
        ssh_user: ubuntu
    accelerator_type: cpu
"""
        config_path = tmp_path / "original.yaml"
        config_path.write_text(config_content)

        # Load config from YAML
        original_config = load_config(config_path)

        # Verify provider is manual before round-trip
        assert original_config.scale_groups["manual_hosts"].HasField("provider")
        provider = original_config.scale_groups["manual_hosts"].provider
        assert provider.WhichOneof("provider") == "manual"
        assert list(provider.manual.hosts) == ["10.0.0.1", "10.0.0.2"]

        # Round-trip: proto → dict → yaml → dict → proto
        config_dict = config_to_dict(original_config)
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        round_trip_path = tmp_path / "round_trip.yaml"
        round_trip_path.write_text(yaml_str)
        loaded_config = load_config(round_trip_path)

        # Verify provider is still manual after round-trip
        assert loaded_config.scale_groups["manual_hosts"].HasField("provider")
        provider_after = loaded_config.scale_groups["manual_hosts"].provider
        assert (
            provider_after.WhichOneof("provider") == "manual"
        ), f"Expected provider to be 'manual' after round-trip, got '{provider_after.WhichOneof('provider')}'"
        assert list(provider_after.manual.hosts) == ["10.0.0.1", "10.0.0.2"]

    def test_multiple_scale_groups_preserve_provider_types(self, tmp_path: Path):
        """Config with mixed TPU and manual providers preserves both types."""
        config_content = """\
provider_type: tpu
project_id: my-project
zone: us-central1-a

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

ssh:
  user: ubuntu
  key_file: ~/.ssh/id_rsa

scale_groups:
  tpu_group:
    provider:
      tpu:
        project_id: my-project
    accelerator_type: tpu
    accelerator_variant: v5litepod-8
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 10
    zones: [us-central1-a]
  manual_group:
    provider:
      manual:
        hosts: [10.0.0.1]
        ssh_user: ubuntu
    accelerator_type: cpu
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

        # Verify TPU group
        tpu_provider = loaded_config.scale_groups["tpu_group"].provider
        assert tpu_provider.WhichOneof("provider") == "tpu"
        assert tpu_provider.tpu.project_id == "my-project"

        # Verify manual group
        manual_provider = loaded_config.scale_groups["manual_group"].provider
        assert manual_provider.WhichOneof("provider") == "manual"
        assert list(manual_provider.manual.hosts) == ["10.0.0.1"]

    def test_example_eu_west4_config_round_trips(self):
        """Real example config from examples/eu-west4.yaml round-trips correctly."""
        iris_root = Path(__file__).parent.parent.parent.parent
        config_path = iris_root / "examples" / "eu-west4.yaml"
        if not config_path.exists():
            pytest.skip("Example config not found")

        original_config = load_config(config_path)

        # Verify it has TPU provider before round-trip
        assert "tpu_v5e_16" in original_config.scale_groups
        provider = original_config.scale_groups["tpu_v5e_16"].provider
        assert provider.WhichOneof("provider") == "tpu"

        # Round-trip via dict and ParseDict
        config_dict = config_to_dict(original_config)
        loaded_config = ParseDict(config_dict, config_pb2.IrisClusterConfig())

        # Verify provider is still TPU
        provider_after = loaded_config.scale_groups["tpu_v5e_16"].provider
        assert (
            provider_after.WhichOneof("provider") == "tpu"
        ), f"Expected provider to be 'tpu' after round-trip, got '{provider_after.WhichOneof('provider')}'"

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
provider_type: tpu
project_id: my-project
zone: us-central1-a

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

scale_groups:
  test_group:
    provider:
      manual:
        hosts: [10.0.0.1]
    accelerator_type: {accelerator_type}
    zones: [local]
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.scale_groups["test_group"].accelerator_type == expected_enum

    def test_uppercase_accelerator_types_still_work(self, tmp_path: Path):
        """Config still accepts uppercase accelerator types for backwards compatibility."""
        config_content = """\
provider_type: tpu
project_id: my-project
zone: us-central1-a

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

scale_groups:
  tpu_group:
    provider:
      tpu:
        project_id: my-project
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
    """Tests for create_autoscaler_from_config factory function."""

    def test_creates_autoscaler_with_tpu_provider(self, tmp_path: Path):
        """create_autoscaler_from_config works with TPU provider config."""
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
    provider:
      tpu:
        project_id: my-project
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
        autoscaler = create_autoscaler_from_config(config, dry_run=True)

        assert autoscaler is not None
        assert "tpu_v5e_8" in autoscaler.groups

    def test_creates_autoscaler_with_manual_provider(self, tmp_path: Path):
        """create_autoscaler_from_config works with manual provider config."""
        config_content = """\
provider_type: manual

bootstrap:
  docker_image: gcr.io/project/iris-worker:latest
  worker_port: 10001
  controller_address: "http://10.0.0.1:10000"

ssh:
  user: ubuntu
  key_file: ~/.ssh/id_rsa

scale_groups:
  manual_hosts:
    provider:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
    accelerator_type: cpu
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        autoscaler = create_autoscaler_from_config(config, dry_run=True)

        assert autoscaler is not None
        assert "manual_hosts" in autoscaler.groups

    def test_creates_autoscaler_after_round_trip(self, tmp_path: Path):
        """create_autoscaler_from_config works after config round-trip."""
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
    provider:
      tpu:
        project_id: my-project
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
        autoscaler = create_autoscaler_from_config(loaded_config, dry_run=True)

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
        """get_ssh_config applies per-group SSH overrides for manual provider."""
        config = config_pb2.IrisClusterConfig(
            ssh=config_pb2.SshConfig(
                user="ubuntu",
                key_file="~/.ssh/cluster_key",
                port=22,
            ),
            scale_groups={
                "manual_group": config_pb2.ScaleGroupConfig(
                    name="manual_group",
                    provider=config_pb2.ProviderConfig(
                        manual=config_pb2.ManualProvider(
                            hosts=["10.0.0.1"],
                            ssh_user="admin",
                            ssh_key_file="~/.ssh/group_key",
                            ssh_port=2222,
                        )
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
                    provider=config_pb2.ProviderConfig(
                        manual=config_pb2.ManualProvider(
                            hosts=["10.0.0.1"],
                            ssh_user="admin",  # Override user only
                        )
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
