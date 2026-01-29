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
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)

# Re-export IrisClusterConfig from proto for public API
IrisClusterConfig = config_pb2.IrisClusterConfig

# Mapping from lowercase accelerator types to proto enum names
_ACCELERATOR_TYPE_MAP = {
    "cpu": "ACCELERATOR_TYPE_CPU",
    "gpu": "ACCELERATOR_TYPE_GPU",
    "tpu": "ACCELERATOR_TYPE_TPU",
}


def _normalize_accelerator_types(data: dict) -> None:
    """Convert lowercase accelerator_type values to proto enum format.

    Modifies data in-place, converting values like "tpu" to "ACCELERATOR_TYPE_TPU".
    This allows YAML configs to use the simpler lowercase format while maintaining
    compatibility with protobuf's enum parsing.
    """
    if "scale_groups" not in data:
        return

    for sg_data in data["scale_groups"].values():
        if sg_data is None:
            continue
        if "accelerator_type" not in sg_data:
            continue

        accel_type = sg_data["accelerator_type"]
        if isinstance(accel_type, str):
            lower_type = accel_type.lower()
            if lower_type in _ACCELERATOR_TYPE_MAP:
                sg_data["accelerator_type"] = _ACCELERATOR_TYPE_MAP[lower_type]


def _validate_accelerator_types(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that scale groups have explicit accelerator types."""
    for name, sg_config in config.scale_groups.items():
        if sg_config.accelerator_type == config_pb2.ACCELERATOR_TYPE_UNSPECIFIED:
            raise ValueError(f"Scale group '{name}' must set accelerator_type to cpu, gpu, or tpu.")


def load_config(config_path: Path | str) -> config_pb2.IrisClusterConfig:
    """Load cluster config from YAML file.

    Configuration uses a nested structure with bootstrap, timeouts, and ssh sub-configs.
    The controller_vm field uses a oneof pattern for type-safe configuration.

    GCP controller example:
    ```yaml
    provider_type: tpu
    project_id: my-project
    region: us-east1
    zone: us-east1-d

    bootstrap:
      docker_image: gcr.io/project/iris-worker:v1
      worker_port: 10001
      controller_address: "10.0.0.1:10000"

    timeouts:
      boot_timeout_seconds: 300
      init_timeout_seconds: 600
      ssh_poll_interval_seconds: 5

    controller_vm:
      gcp:
        image: gcr.io/project/iris-controller:v1
        machine_type: n2-standard-4
        port: 10000

    scale_groups:
      tpu_v5p_8:
        provider:
          tpu:
            project_id: my-project
        accelerator_type: tpu
        accelerator_variant: v5p-8
        runtime_version: v2-alpha-tpuv5
        zones: [us-central1-a]
        min_slices: 0
        max_slices: 10
    ```

    Manual controller example:
    ```yaml
    provider_type: manual

    bootstrap:
      docker_image: gcr.io/project/iris-worker:v1
      worker_port: 10001

    ssh:
      user: ubuntu
      key_file: ~/.ssh/key.pem
      port: 22
      connect_timeout: 30

    controller_vm:
      manual:
        host: 10.0.0.100
        image: gcr.io/project/iris-controller:v1
        port: 10000

    scale_groups:
      manual_hosts:
        provider:
          manual:
            hosts: [10.0.0.1, 10.0.0.2]
            ssh_user: ubuntu        # Per-group SSH override
            ssh_key_file: ~/.ssh/manual_key
    ```
    """
    config_path = Path(config_path)
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    # Expand environment variables in controller_address only.
    # Other fields (e.g., docker_image, ssh.key_file) are used as-is.
    # This is intentional - controller_address often needs $IRIS_CONTROLLER_ADDRESS for dynamic discovery.
    if "bootstrap" in data and "controller_address" in data["bootstrap"]:
        data["bootstrap"]["controller_address"] = os.path.expandvars(data["bootstrap"]["controller_address"])

    # Ensure scale_groups have their name field set (proto uses map key, but config field needs it)
    if "scale_groups" in data:
        for name, sg_data in data["scale_groups"].items():
            if sg_data is None:
                data["scale_groups"][name] = {"name": name}
            elif "name" not in sg_data:
                sg_data["name"] = name

    # Convert lowercase accelerator types to enum format
    _normalize_accelerator_types(data)

    config = ParseDict(data, config_pb2.IrisClusterConfig())
    _validate_accelerator_types(config)

    logger.info(
        "Config loaded: provider=%s, scale_groups=%s",
        config.provider_type or "manual",
        list(config.scale_groups.keys()) if config.scale_groups else "(none)",
    )

    return config


def config_to_dict(config: config_pb2.IrisClusterConfig) -> dict:
    """Convert config to dict for YAML serialization."""
    return MessageToDict(config, preserving_proto_field_name=True)


def get_ssh_config(
    cluster_config: config_pb2.IrisClusterConfig,
    group_name: str | None = None,
) -> SshConfig:
    """Get SSH config by merging cluster defaults with per-group overrides.

    If cluster_config.ssh is not set, uses defaults:
    - user: "root"
    - port: 22
    - connect_timeout: 30
    - key_file: None (passwordless/agent auth)

    For manual providers, per-group overrides from provider.manual take precedence.

    Args:
        cluster_config: The cluster configuration.
        group_name: Optional scale group name. If provided and the group uses
            manual provider, per-group SSH overrides will be applied.

    Returns:
        SshConfig with all settings populated (using defaults where not specified).
    """
    ssh = cluster_config.ssh
    user = ssh.user or "root"
    key_file = ssh.key_file or None
    port = ssh.port or 22
    connect_timeout = ssh.connect_timeout or 30

    # Apply per-group overrides if group_name provided
    if group_name and group_name in cluster_config.scale_groups:
        group_config = cluster_config.scale_groups[group_name]
        if group_config.HasField("provider"):
            provider = group_config.provider
            if provider.WhichOneof("provider") == "manual":
                manual = provider.manual
                if manual.ssh_user:
                    user = manual.ssh_user
                if manual.ssh_key_file:
                    key_file = manual.ssh_key_file
                if manual.ssh_port:
                    port = manual.ssh_port

    return SshConfig(
        user=user,
        key_file=key_file,
        port=port,
        connect_timeout=connect_timeout,
    )


def with_timeout_defaults(timeouts: config_pb2.TimeoutConfig) -> config_pb2.TimeoutConfig:
    """Apply default values to unset timeout fields.

    Protobuf uses 0 as the default for int32 fields, which can cause issues
    when 0 is passed to code expecting positive values. This function ensures
    all timeout fields have sensible defaults.
    """
    return config_pb2.TimeoutConfig(
        boot_timeout_seconds=timeouts.boot_timeout_seconds or 300,
        init_timeout_seconds=timeouts.init_timeout_seconds or 600,
        ssh_poll_interval_seconds=timeouts.ssh_poll_interval_seconds or 5,
    )


@dataclass
class ScaleGroupSpec:
    """Extended scale group specification with provider info.

    Wraps a ScaleGroupConfig proto with additional metadata needed for
    factory instantiation, such as the provider type and manual hosts.
    """

    config: config_pb2.ScaleGroupConfig
    provider: str = "tpu"
    hosts: list[str] = field(default_factory=list)


def _get_provider_info(
    group_config: config_pb2.ScaleGroupConfig,
) -> tuple[str, str | None, list[str] | None]:
    """Extract provider type and provider-specific info from scale group.

    Returns: (provider_type, project_id, hosts)
        - For TPU: (\"tpu\", project_id, None)
        - For manual: (\"manual\", None, hosts)

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
        return ("tpu", provider.tpu.project_id, None)
    if which == "manual":
        manual = provider.manual
        if not manual.hosts:
            raise ValueError(f"Manual provider in {group_config.name} missing hosts")
        return ("manual", None, list(manual.hosts))
    if which == "local":
        return ("local", None, None)

    raise ValueError(f"Unknown provider type in scale group {group_config.name}")


def create_autoscaler_from_config(
    config: config_pb2.IrisClusterConfig,
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

    for name in config.scale_groups:
        manager = _create_manager_from_config(
            group_name=name,
            cluster_config=config,
            vm_factory=vm_factory,
            dry_run=dry_run,
        )

        scale_groups[name] = ScalingGroup(
            config=config.scale_groups[name],
            vm_manager=manager,
        )

        logger.info("Created scale group %s", name)

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=autoscaler_config or AutoscalerConfig(),
    )


def create_autoscaler_from_specs(
    specs: dict[str, ScaleGroupSpec],
    project_id: str,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeouts: config_pb2.TimeoutConfig,
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
    bootstrap_config = config_pb2.BootstrapConfig(
        controller_address=controller_address,
        docker_image=docker_image,
        worker_port=worker_port,
    )

    ssh_config = SshConfig(
        user=ssh_user,
        key_file=ssh_key,
    )

    sg_config = config_pb2.ScaleGroupConfig(
        name="manual",
        min_slices=0,
        max_slices=len(hosts),
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        zones=["local"],
    )

    timeouts = config_pb2.TimeoutConfig(
        boot_timeout_seconds=300,
        init_timeout_seconds=600,
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


def _create_manager_from_config(
    group_name: str,
    cluster_config: config_pb2.IrisClusterConfig,
    vm_factory: TrackedVmFactory,
    *,
    dry_run: bool = False,
) -> VmManagerProtocol:
    """Create a VmManager for a scale group from config.

    Args:
        group_name: Name of the scale group in cluster_config.scale_groups
        cluster_config: The full cluster configuration
        vm_factory: Factory for creating VMs
        dry_run: If True, don't actually create VMs

    Returns:
        A VmManager appropriate for the provider type

    Raises:
        ValueError: If group_name not found, provider missing, or invalid config
    """
    if group_name not in cluster_config.scale_groups:
        raise ValueError(f"Scale group {group_name} not found in cluster config")

    group_config = cluster_config.scale_groups[group_name]
    provider_type, project_id, hosts = _get_provider_info(group_config)

    label_prefix = cluster_config.label_prefix or "iris"
    timeouts = with_timeout_defaults(cluster_config.timeouts)

    if provider_type == "tpu":
        if not project_id:
            raise ValueError(f"project_id required for TPU scale group {group_name}")
        return TpuVmManager(  # type: ignore[return-value]
            project_id=project_id,
            config=group_config,
            bootstrap_config=cluster_config.bootstrap,
            timeouts=timeouts,
            vm_factory=vm_factory,
            label_prefix=label_prefix,
            dry_run=dry_run,
        )

    if provider_type == "manual":
        if not hosts:
            raise ValueError(f"hosts required for manual scale group {group_name}")
        ssh_config = get_ssh_config(cluster_config, group_name)
        return ManualVmManager(
            hosts=hosts,
            config=group_config,
            bootstrap_config=cluster_config.bootstrap,
            timeouts=timeouts,
            vm_factory=vm_factory,
            ssh_config=ssh_config,
            label_prefix=label_prefix,
            dry_run=dry_run,
        )

    raise ValueError(f"Unknown provider: {provider_type}")


def _create_manager(
    provider: str,
    config: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeouts: config_pb2.TimeoutConfig,
    vm_factory: TrackedVmFactory,
    *,
    project_id: str | None = None,
    hosts: list[str] | None = None,
    ssh_config: SshConfig | None = None,
    label_prefix: str = "iris",
    dry_run: bool = False,
) -> VmManagerProtocol:
    """Create the appropriate VmManager based on provider type.

    This is the lower-level factory used by create_autoscaler_from_specs
    when the caller provides explicit bootstrap/timeout/ssh configs rather
    than resolving from a cluster config.
    """
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
