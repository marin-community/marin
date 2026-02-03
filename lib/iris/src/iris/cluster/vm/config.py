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
from iris.cluster.vm.platform import create_platform
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.cluster.vm.vm_platform import VmManagerProtocol
from iris.rpc import config_pb2, time_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)

# Re-export IrisClusterConfig from proto for public API
IrisClusterConfig = config_pb2.IrisClusterConfig

# Mapping from lowercase accelerator types to proto enum names
_ACCELERATOR_TYPE_MAP = {
    "cpu": "ACCELERATOR_TYPE_CPU",
    "gpu": "ACCELERATOR_TYPE_GPU",
    "tpu": "ACCELERATOR_TYPE_TPU",
}

_VM_TYPE_MAP = {
    "tpu_vm": "VM_TYPE_TPU_VM",
    "gce_vm": "VM_TYPE_GCE_VM",
    "manual_vm": "VM_TYPE_MANUAL_VM",
    "local_vm": "VM_TYPE_LOCAL_VM",
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


def _normalize_vm_types(data: dict) -> None:
    """Convert lowercase vm_type values to proto enum format."""
    if "scale_groups" not in data:
        return

    for sg_data in data["scale_groups"].values():
        if sg_data is None:
            continue
        if "vm_type" not in sg_data:
            continue

        vm_type = sg_data["vm_type"]
        if isinstance(vm_type, str):
            lower_type = vm_type.lower()
            if lower_type in _VM_TYPE_MAP:
                sg_data["vm_type"] = _VM_TYPE_MAP[lower_type]


def _validate_accelerator_types(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that scale groups have explicit accelerator types."""
    for name, sg_config in config.scale_groups.items():
        if sg_config.accelerator_type == config_pb2.ACCELERATOR_TYPE_UNSPECIFIED:
            raise ValueError(f"Scale group '{name}' must set accelerator_type to cpu, gpu, or tpu.")


def _validate_vm_types(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that scale groups have explicit vm_type."""
    for name, sg_config in config.scale_groups.items():
        if sg_config.vm_type == config_pb2.VM_TYPE_UNSPECIFIED:
            raise ValueError(f"Scale group '{name}' must set vm_type to tpu_vm, gce_vm, manual_vm, or local_vm.")


def _resolve_duration(
    primary: config_pb2.TimeoutConfig | config_pb2.AutoscalerConfig,
    fallback: config_pb2.TimeoutConfig | config_pb2.AutoscalerConfig | config_pb2.AutoscalerDefaults,
    field_name: str,
    default_value: Duration,
) -> time_pb2.Duration:
    if primary.HasField(field_name):
        field = getattr(primary, field_name)
        if field.milliseconds > 0:
            return field
    if fallback.HasField(field_name):
        field = getattr(fallback, field_name)
        if field.milliseconds > 0:
            return field
    return default_value.to_proto()


def _merge_timeouts(
    timeouts: config_pb2.TimeoutConfig,
    defaults: config_pb2.TimeoutConfig,
) -> config_pb2.TimeoutConfig:
    return config_pb2.TimeoutConfig(
        boot_timeout=_resolve_duration(timeouts, defaults, "boot_timeout", Duration.from_seconds(300)),
        init_timeout=_resolve_duration(timeouts, defaults, "init_timeout", Duration.from_seconds(600)),
        ssh_poll_interval=_resolve_duration(timeouts, defaults, "ssh_poll_interval", Duration.from_seconds(5)),
    )


def _merge_autoscaler(
    autoscaler: config_pb2.AutoscalerConfig,
    defaults: config_pb2.AutoscalerDefaults,
) -> config_pb2.AutoscalerConfig:
    return config_pb2.AutoscalerConfig(
        evaluation_interval=_resolve_duration(
            autoscaler,
            defaults,
            "evaluation_interval",
            Duration.from_seconds(10),
        ),
        requesting_timeout=_resolve_duration(
            autoscaler,
            defaults,
            "requesting_timeout",
            Duration.from_seconds(120),
        ),
        scale_up_delay=_resolve_duration(
            autoscaler,
            defaults,
            "scale_up_delay",
            Duration.from_seconds(60),
        ),
        scale_down_delay=_resolve_duration(
            autoscaler,
            defaults,
            "scale_down_delay",
            Duration.from_seconds(300),
        ),
    )


def _merge_ssh(
    ssh: config_pb2.SshConfig,
    defaults: config_pb2.SshConfig,
) -> config_pb2.SshConfig:
    connect_timeout = ssh.connect_timeout
    if not (ssh.HasField("connect_timeout") and ssh.connect_timeout.milliseconds > 0):
        connect_timeout = defaults.connect_timeout
    if not (connect_timeout.milliseconds > 0):
        connect_timeout = Duration.from_seconds(30).to_proto()

    return config_pb2.SshConfig(
        user=ssh.user or defaults.user or "root",
        key_file=ssh.key_file or defaults.key_file,
        port=ssh.port or defaults.port or 22,
        connect_timeout=connect_timeout,
    )


def _merge_bootstrap(
    bootstrap: config_pb2.BootstrapConfig,
    defaults: config_pb2.BootstrapConfig,
) -> config_pb2.BootstrapConfig:
    env_vars: dict[str, str] = {}
    env_vars.update(defaults.env_vars)
    env_vars.update(bootstrap.env_vars)

    worker_port = bootstrap.worker_port or defaults.worker_port or 10001
    cache_dir = bootstrap.cache_dir or defaults.cache_dir or "/var/cache/iris"

    return config_pb2.BootstrapConfig(
        controller_address=bootstrap.controller_address or defaults.controller_address,
        worker_id=bootstrap.worker_id or defaults.worker_id,
        worker_port=worker_port,
        docker_image=bootstrap.docker_image or defaults.docker_image,
        cache_dir=cache_dir,
        env_vars=env_vars,
    )


def apply_defaults(config: config_pb2.IrisClusterConfig) -> config_pb2.IrisClusterConfig:
    """Apply defaults to a config and return a new merged config."""
    merged = config_pb2.IrisClusterConfig()
    merged.CopyFrom(config)

    defaults = config.defaults if config.HasField("defaults") else config_pb2.DefaultsConfig()

    merged.timeouts.CopyFrom(_merge_timeouts(config.timeouts, defaults.timeouts))
    merged.ssh.CopyFrom(_merge_ssh(config.ssh, defaults.ssh))
    merged.autoscaler.CopyFrom(_merge_autoscaler(config.autoscaler, defaults.autoscaler))
    merged.bootstrap.CopyFrom(_merge_bootstrap(config.bootstrap, defaults.bootstrap))

    for group in merged.scale_groups.values():
        if not group.HasField("priority"):
            group.priority = 100

    return merged


def load_config(config_path: Path | str) -> config_pb2.IrisClusterConfig:
    """Load cluster config from YAML file."""
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
    if "defaults" in data and "bootstrap" in data["defaults"]:
        defaults_bootstrap = data["defaults"]["bootstrap"]
        if "controller_address" in defaults_bootstrap:
            defaults_bootstrap["controller_address"] = os.path.expandvars(defaults_bootstrap["controller_address"])

    if isinstance(data.get("platform"), str):
        data["platform"] = {data["platform"]: {}}

    # Ensure scale_groups have their name field set (proto uses map key, but config field needs it)
    if "scale_groups" in data:
        for name, sg_data in data["scale_groups"].items():
            if sg_data is None:
                data["scale_groups"][name] = {"name": name}
                sg_data = data["scale_groups"][name]
            elif "name" not in sg_data:
                sg_data["name"] = name
            if "vm_type" not in sg_data and "type" in sg_data:
                sg_data["vm_type"] = sg_data.pop("type")

    # Convert lowercase accelerator types to enum format
    _normalize_accelerator_types(data)
    _normalize_vm_types(data)

    config = ParseDict(data, config_pb2.IrisClusterConfig())
    config = apply_defaults(config)
    _validate_accelerator_types(config)
    _validate_vm_types(config)

    platform_kind = config.platform.WhichOneof("platform") if config.HasField("platform") else "unspecified"
    logger.info(
        "Config loaded: platform=%s, scale_groups=%s",
        platform_kind,
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
    - connect_timeout: 30s
    - key_file: None (passwordless/agent auth)

    For manual providers, per-group overrides from scale_groups[*].manual take precedence.

    Args:
        cluster_config: The cluster configuration.
        group_name: Optional scale group name. If provided and the group uses
            manual VM type, per-group SSH overrides will be applied.

    Returns:
        SshConfig with all settings populated (using defaults where not specified).
    """
    from iris.time_utils import Duration

    ssh = cluster_config.ssh
    user = ssh.user or "root"
    key_file = ssh.key_file or None
    port = ssh.port or 22
    connect_timeout = (
        Duration.from_proto(ssh.connect_timeout)
        if ssh.HasField("connect_timeout") and ssh.connect_timeout.milliseconds > 0
        else Duration.from_seconds(30)
    )

    # Apply per-group overrides if group_name provided
    if group_name and group_name in cluster_config.scale_groups:
        group_config = cluster_config.scale_groups[group_name]
        if group_config.manual.hosts:
            manual = group_config.manual
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


@dataclass
class ScaleGroupSpec:
    """Extended scale group specification with provider info.

    Wraps a ScaleGroupConfig proto with additional metadata needed for
    factory instantiation, such as the provider type and manual hosts.
    """

    config: config_pb2.ScaleGroupConfig
    provider: str = "tpu"
    hosts: list[str] = field(default_factory=list)


def create_autoscaler_from_config(
    config: config_pb2.IrisClusterConfig,
    autoscaler_config: config_pb2.AutoscalerConfig | None = None,
    dry_run: bool = False,
):
    """Create autoscaler with per-group managers from configuration.

    This is the main entry point for creating a production autoscaler.
    It creates:
    - A shared VmRegistry for global VM tracking
    - A TrackedVmFactory that registers VMs automatically
    - A VmManager per scale group via the platform
    - ScalingGroups that own VM groups and track scaling state
    - The Autoscaler that coordinates scaling decisions

    Args:
        config: Cluster configuration proto with scale groups
        autoscaler_config: Optional autoscaler configuration proto (overrides config.autoscaler if provided)
        dry_run: If True, don't actually create VMs

    Returns:
        A fully configured Autoscaler ready for use

    Raises:
        ValueError: If a scale group is missing vm_type or platform config is invalid
    """
    from iris.cluster.vm.autoscaler import Autoscaler

    config = apply_defaults(config)
    logger.info("Creating Autoscaler")

    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)
    platform = create_platform(config)

    scale_groups: dict[str, ScalingGroup] = {}
    if autoscaler_config is None:
        autoscaler_config = config_pb2.AutoscalerConfig(
            scale_up_delay=Duration.from_seconds(60).to_proto(),
            scale_down_delay=Duration.from_seconds(300).to_proto(),
        )
    scale_up_delay = Duration.from_proto(autoscaler_config.scale_up_delay)
    scale_down_delay = Duration.from_proto(autoscaler_config.scale_down_delay)
    scale_up_delay = Duration.from_proto(config.autoscaler.scale_up_delay)
    scale_down_delay = Duration.from_proto(config.autoscaler.scale_down_delay)

    for name, group_config in config.scale_groups.items():
        manager = platform.vm_manager(group_config, vm_factory=vm_factory, dry_run=dry_run)
        scale_groups[name] = ScalingGroup(
            config=group_config,
            vm_manager=manager,
            scale_up_cooldown=scale_up_delay,
            scale_down_cooldown=scale_down_delay,
        )
        logger.info("Created scale group %s", name)

    if autoscaler_config is None:
        autoscaler_config = config.autoscaler

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=autoscaler_config,
    )


def create_autoscaler_from_specs(
    specs: dict[str, ScaleGroupSpec],
    project_id: str,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeouts: config_pb2.TimeoutConfig,
    ssh_config: SshConfig | None = None,
    autoscaler_config: config_pb2.AutoscalerConfig | None = None,
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
        autoscaler_config: Optional autoscaler configuration proto
        label_prefix: Prefix for GCP labels

    Returns:
        A fully configured Autoscaler ready for use

    Raises:
        ValueError: If a scale group has an unknown provider type
    """
    from iris.cluster.vm.autoscaler import Autoscaler

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
            scale_up_cooldown=scale_up_delay,
            scale_down_cooldown=scale_down_delay,
        )

        logger.info(
            "Created scale group %s with provider=%s",
            name,
            spec.provider,
        )

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=autoscaler_config,
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
        vm_type=config_pb2.VM_TYPE_MANUAL_VM,
        min_slices=0,
        max_slices=len(hosts),
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        zones=["local"],
    )

    timeouts = config_pb2.TimeoutConfig()
    timeouts.boot_timeout.CopyFrom(Duration.from_seconds(300).to_proto())
    timeouts.init_timeout.CopyFrom(Duration.from_seconds(600).to_proto())
    timeouts.ssh_poll_interval.CopyFrom(Duration.from_seconds(5).to_proto())

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
