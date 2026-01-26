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

"""Configuration loading and autoscaler factory for VM CLI.

Supports both YAML config files (for full cluster management) and
programmatic configuration (for quick CLI flag-based operations).

This module provides the main entry points for creating autoscalers:
- create_autoscaler_from_config: Create from IrisClusterConfig proto
- create_autoscaler_from_specs: Create from explicit ScaleGroupSpec list
- create_manual_autoscaler: Quick path for manual hosts without config file
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from google.protobuf.json_format import MessageToDict, ParseDict

from iris.cluster.vm.gcp_tpu_platform import TpuVmManager
from iris.cluster.vm.managed_vm import SshConfig, TrackedVmFactory, VmRegistry
from iris.cluster.vm.manual_platform import ManualVmManager
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.cluster.vm.vm_platform import VmManagerProtocol
from iris.rpc import vm_pb2

logger = logging.getLogger(__name__)

# Re-export IrisClusterConfig from proto for backwards compatibility with __init__.py
IrisClusterConfig = vm_pb2.IrisClusterConfig


def load_config(config_path: Path | str) -> vm_pb2.IrisClusterConfig:
    """Load cluster config from YAML file.

    YAML structure uses flat field names matching the IrisClusterConfig proto:

    ```yaml
    provider_type: tpu
    project_id: my-project
    region: us-east1
    zone: us-east1-d

    ssh_user: ray
    ssh_private_key: ~/.ssh/key.pem

    docker_image: gcr.io/project/iris-worker:v1
    worker_port: 10001

    controller_address: "10.0.0.1:10000"
    controller_vm:
      enabled: true
      image: gcr.io/project/iris-controller:v1
      port: 10000

    manual_hosts:
      - 10.0.0.1
      - 10.0.0.2

    scale_groups:
      tpu_v5p_8:
        accelerator_type: v5p-8
        runtime_version: v2-alpha-tpuv5
        zones: [us-central1-a]
        min_slices: 0
        max_slices: 10
        preemptible: true
    ```
    """
    config_path = Path(config_path)
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Expand env vars in controller_address
    if "controller_address" in data:
        data["controller_address"] = os.path.expandvars(data["controller_address"])

    # Ensure scale_groups have their name field set (proto uses map key, but config field needs it)
    if "scale_groups" in data:
        for name, sg_data in data["scale_groups"].items():
            if sg_data is None:
                data["scale_groups"][name] = {"name": name}
            elif "name" not in sg_data:
                sg_data["name"] = name

    config = ParseDict(data, vm_pb2.IrisClusterConfig())

    # Warn about missing controller address only for manual provider without controller VM.
    if not config.controller_address and not config.controller_vm.enabled and config.provider_type == "manual":
        logger.warning("No controller address configured - workers will fail to start")

    logger.info(
        "Config loaded: provider=%s, scale_groups=%s",
        config.provider_type or "manual",
        list(config.scale_groups.keys()) if config.scale_groups else "(none)",
    )

    return config


def config_to_dict(config: vm_pb2.IrisClusterConfig) -> dict:
    """Convert config to dict for YAML serialization."""
    return MessageToDict(config, preserving_proto_field_name=True)


def to_bootstrap_config(config: vm_pb2.IrisClusterConfig) -> vm_pb2.BootstrapConfig:
    """Convert cluster config to BootstrapConfig proto."""
    return vm_pb2.BootstrapConfig(
        controller_address=config.controller_address,
        docker_image=config.docker_image,
        worker_port=config.worker_port,
    )


def to_timeout_config(config: vm_pb2.IrisClusterConfig) -> vm_pb2.TimeoutConfig:
    """Convert cluster config to TimeoutConfig proto with defaults."""
    return vm_pb2.TimeoutConfig(
        boot_timeout_seconds=config.boot_timeout_seconds or 300,
        init_timeout_seconds=config.init_timeout_seconds or 600,
        ssh_connect_timeout_seconds=config.ssh_connect_timeout_seconds or 30,
        ssh_poll_interval_seconds=config.ssh_poll_interval_seconds or 5,
    )


def to_ssh_config(config: vm_pb2.IrisClusterConfig) -> SshConfig:
    """Convert cluster config to SshConfig for manual SSH."""
    return SshConfig(
        user=config.ssh_user or "root",
        key_file=config.ssh_private_key or None,
        connect_timeout=config.ssh_connect_timeout_seconds or 30,
    )


@dataclass
class ScaleGroupSpec:
    """Extended scale group specification with provider info.

    Wraps a ScaleGroupConfig proto with additional metadata needed for
    factory instantiation, such as the provider type and manual hosts.
    """

    config: vm_pb2.ScaleGroupConfig
    provider: str = "tpu"
    hosts: list[str] = field(default_factory=list)


def _get_provider_config(
    group_config: vm_pb2.ScaleGroupConfig,
    cluster_config: vm_pb2.IrisClusterConfig,
) -> tuple[str, str | None, list[str] | None, SshConfig | None]:
    """Extract provider type and config from scale group.

    Returns: (provider_type, project_id, hosts, ssh_config)

    Raises:
        ValueError: If scale group missing provider config or unknown provider type
    """
    if not group_config.HasField("provider"):
        raise ValueError(f"Scale group {group_config.name} missing provider config")

    provider = group_config.provider
    which = provider.WhichOneof("provider")
    if which == "tpu":
        if not provider.tpu.project_id:
            raise ValueError(f"TPU provider in {group_config.name} missing project_id")
        return ("tpu", provider.tpu.project_id, None, None)
    if which == "manual":
        manual = provider.manual
        if not manual.hosts:
            raise ValueError(f"Manual provider in {group_config.name} missing hosts")
        ssh_config = SshConfig(
            user=manual.ssh_user or cluster_config.ssh_user or "root",
            key_file=manual.ssh_key_file or cluster_config.ssh_private_key or None,
            connect_timeout=cluster_config.ssh_connect_timeout_seconds or 30,
            port=manual.ssh_port or 22,
        )
        return ("manual", None, list(manual.hosts), ssh_config)

    raise ValueError(f"Unknown provider type in scale group {group_config.name}")


def create_autoscaler_from_config(
    config: vm_pb2.IrisClusterConfig,
    autoscaler_config=None,  # AutoscalerConfig | None - type is from autoscaler module
    dry_run: bool = False,
):
    """Create autoscaler with per-group managers from configuration.

    This is the main entry point for creating a production autoscaler.
    It creates:
    - A shared VmRegistry for global VM tracking
    - A TrackedVmFactory that registers VMs automatically
    - A VmManager for each scale group based on its provider
    - ScalingGroups that own VM groups and track scaling state
    - The Autoscaler that coordinates scaling decisions

    Each scale group must have a `provider` field specifying either:
    - `tpu`: with `project_id`
    - `manual`: with `hosts` list

    Args:
        config: Cluster configuration proto with scale groups
        autoscaler_config: Optional autoscaler configuration
        dry_run: If True, don't actually create VMs

    Returns:
        A fully configured Autoscaler ready for use

    Raises:
        ValueError: If a scale group is missing provider config or has unknown provider type
    """
    from iris.cluster.vm.autoscaler import Autoscaler, AutoscalerConfig

    logger.info("Creating Autoscaler")

    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)

    scale_groups: dict[str, ScalingGroup] = {}

    for name, group_config in config.scale_groups.items():
        provider_type, project_id, hosts, ssh_config = _get_provider_config(group_config, config)

        # Use per-group ssh_config if available, otherwise fall back to cluster-level
        if ssh_config is None and provider_type == "manual":
            ssh_config = to_ssh_config(config)

        manager = _create_manager(
            provider=provider_type,
            config=group_config,
            bootstrap_config=to_bootstrap_config(config),
            timeouts=to_timeout_config(config),
            vm_factory=vm_factory,
            project_id=project_id,
            hosts=hosts,
            ssh_config=ssh_config,
            label_prefix=config.label_prefix or "iris",
            dry_run=dry_run,
        )

        scale_groups[name] = ScalingGroup(
            config=group_config,
            vm_manager=manager,
        )

        logger.info(
            "Created scale group %s with provider=%s",
            name,
            provider_type,
        )

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=autoscaler_config or AutoscalerConfig(),
    )


def create_autoscaler_from_specs(
    specs: dict[str, ScaleGroupSpec],
    project_id: str,
    bootstrap_config: vm_pb2.BootstrapConfig,
    timeouts: vm_pb2.TimeoutConfig,
    ssh_config: SshConfig | None = None,
    autoscaler_config=None,
    label_prefix: str = "iris",
    dry_run: bool = False,
):
    """Create autoscaler from explicit scale group specs.

    This is useful when you have fine-grained control over each group's
    provider type and configuration, rather than using a unified config file.

    Args:
        specs: Map of scale group name to ScaleGroupSpec
        project_id: GCP project ID (for TPU groups)
        bootstrap_config: Bootstrap configuration for all VMs
        timeouts: Timeout configuration for all VMs
        ssh_config: SSH configuration for manual groups
        autoscaler_config: Optional autoscaler configuration
        label_prefix: Prefix for GCP labels

    Returns:
        A fully configured Autoscaler ready for use

    Raises:
        ValueError: If a scale group has an unknown provider type
    """
    from iris.cluster.vm.autoscaler import Autoscaler, AutoscalerConfig

    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)

    scale_groups: dict[str, ScalingGroup] = {}

    for name, spec in specs.items():
        manager = _create_manager(
            provider=spec.provider,
            config=spec.config,
            bootstrap_config=bootstrap_config,
            timeouts=timeouts,
            vm_factory=vm_factory,
            project_id=project_id,
            hosts=spec.hosts,
            ssh_config=ssh_config,
            label_prefix=label_prefix,
            dry_run=dry_run,
        )

        scale_groups[name] = ScalingGroup(
            config=spec.config,
            vm_manager=manager,
        )

        logger.info(
            "Created scale group %s with provider=%s",
            name,
            spec.provider,
        )

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=autoscaler_config or AutoscalerConfig(),
    )


def create_manual_autoscaler(
    hosts: list[str],
    controller_address: str,
    docker_image: str,
    ssh_user: str = "root",
    ssh_key: str | None = None,
    worker_port: int = 10001,
):
    """Create a ManualVmManager-based Autoscaler directly from CLI flags.

    This is the quick path for initializing hosts without a config file.
    Builds appropriate spec objects and delegates to create_autoscaler_from_specs.
    """
    bootstrap_config = vm_pb2.BootstrapConfig(
        controller_address=controller_address,
        docker_image=docker_image,
        worker_port=worker_port,
    )

    ssh_config = SshConfig(
        user=ssh_user,
        key_file=ssh_key,
    )

    sg_config = vm_pb2.ScaleGroupConfig(
        name="manual",
        min_slices=0,
        max_slices=len(hosts),
        accelerator_type="cpu",
        zones=["local"],
    )

    timeouts = vm_pb2.TimeoutConfig(
        boot_timeout_seconds=300,
        init_timeout_seconds=600,
        ssh_connect_timeout_seconds=30,
        ssh_poll_interval_seconds=5,
    )

    spec = ScaleGroupSpec(
        config=sg_config,
        provider="manual",
        hosts=hosts,
    )

    return create_autoscaler_from_specs(
        specs={"manual": spec},
        project_id="",
        bootstrap_config=bootstrap_config,
        timeouts=timeouts,
        ssh_config=ssh_config,
    )


def _create_manager(
    provider: str,
    config: vm_pb2.ScaleGroupConfig,
    bootstrap_config: vm_pb2.BootstrapConfig,
    timeouts: vm_pb2.TimeoutConfig,
    vm_factory: TrackedVmFactory,
    *,
    project_id: str | None = None,
    hosts: list[str] | None = None,
    ssh_config: SshConfig | None = None,
    label_prefix: str = "iris",
    dry_run: bool = False,
) -> VmManagerProtocol:
    """Create the appropriate VmManager based on provider type."""
    if provider == "tpu":
        if not project_id:
            raise ValueError(f"project_id required for TPU scale group {config.name}")
        return TpuVmManager(  # type: ignore[return-value]
            project_id=project_id,
            config=config,
            bootstrap_config=bootstrap_config,
            timeouts=timeouts,
            vm_factory=vm_factory,
            label_prefix=label_prefix,
            dry_run=dry_run,
        )

    if provider == "manual":
        if not hosts:
            raise ValueError(f"hosts required for manual scale group {config.name}")
        return ManualVmManager(
            hosts=hosts,
            config=config,
            bootstrap_config=bootstrap_config,
            timeouts=timeouts,
            vm_factory=vm_factory,
            ssh_config=ssh_config,
            label_prefix=label_prefix,
            dry_run=dry_run,
        )

    raise ValueError(f"Unknown provider: {provider}")
