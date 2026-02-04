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

from iris.cluster.types import parse_memory_string
from iris.cluster.vm.gcp_tpu_platform import TpuVmManager
from iris.cluster.vm.managed_vm import SshConfig, TrackedVmFactory, VmRegistry
from iris.cluster.vm.manual_platform import ManualVmManager
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.cluster.vm.vm_platform import VmManagerProtocol
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)

DEFAULT_SSH_PORT = 22

# Re-export IrisClusterConfig from proto for public API
IrisClusterConfig = config_pb2.IrisClusterConfig

# Single source of truth for all default values
DEFAULT_CONFIG = config_pb2.DefaultsConfig(
    timeouts=config_pb2.TimeoutConfig(
        boot_timeout=Duration.from_seconds(300).to_proto(),
        init_timeout=Duration.from_seconds(600).to_proto(),
        ssh_poll_interval=Duration.from_seconds(5).to_proto(),
    ),
    ssh=config_pb2.SshConfig(
        user="root",
        connect_timeout=Duration.from_seconds(30).to_proto(),
    ),
    autoscaler=config_pb2.AutoscalerConfig(
        evaluation_interval=Duration.from_seconds(10).to_proto(),
        requesting_timeout=Duration.from_seconds(120).to_proto(),
        scale_up_delay=Duration.from_seconds(60).to_proto(),
        scale_down_delay=Duration.from_seconds(300).to_proto(),
    ),
    bootstrap=config_pb2.BootstrapConfig(
        worker_port=10001,
        cache_dir="/var/cache/iris",
    ),
)

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


def _validate_scale_group_resources(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that scale groups define per-VM resources and slice_size."""
    for name, sg_config in config.scale_groups.items():
        if not sg_config.HasField("resources"):
            raise ValueError(f"Scale group '{name}' must set resources.")
        if not sg_config.HasField("slice_size"):
            raise ValueError(f"Scale group '{name}' must set slice_size.")
        if sg_config.slice_size <= 0:
            raise ValueError(f"Scale group '{name}' has invalid slice_size={sg_config.slice_size}.")

        resources = sg_config.resources
        if resources.cpu < 0:
            raise ValueError(f"Scale group '{name}' has invalid cpu={resources.cpu}.")
        if resources.memory_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid memory_bytes={resources.memory_bytes}.")
        if resources.disk_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid disk_bytes={resources.disk_bytes}.")
        if resources.gpu_count < 0:
            raise ValueError(f"Scale group '{name}' has invalid gpu_count={resources.gpu_count}.")
        if resources.tpu_chips < 0:
            raise ValueError(f"Scale group '{name}' has invalid tpu_chips={resources.tpu_chips}.")


def _scale_groups_to_config(scale_groups: dict[str, config_pb2.ScaleGroupConfig]) -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    for name, sg_config in scale_groups.items():
        config.scale_groups[name].CopyFrom(sg_config)
    return config


def _merge_proto_fields(target, source) -> None:
    """Merge explicitly-set fields from source into target.

    With EXPLICIT field presence (proto edition 2023), HasField returns True
    only for fields that were explicitly set by the user. This function trusts
    HasField and copies all explicitly-set values, even zeros/empty strings.

    Validation of invalid values (e.g., zero timeouts) should happen separately
    via explicit validation functions, not silently during merging.

    Note: Map fields (like env_vars) don't support HasField and must be handled
    separately by the caller.

    Args:
        target: Proto message to merge into (modified in place)
        source: Proto message to merge from
    """
    for field_desc in source.DESCRIPTOR.fields:
        field_name = field_desc.name

        # Skip map fields - they don't support HasField
        if field_desc.message_type and field_desc.message_type.GetOptions().map_entry:
            continue

        # With EXPLICIT field presence, HasField is sufficient - trust it
        if not source.HasField(field_name):
            continue

        value = getattr(source, field_name)

        # For message types (Duration, nested messages), use CopyFrom
        if hasattr(value, "CopyFrom"):
            target_field = getattr(target, field_name)
            target_field.CopyFrom(value)
        # For scalar types (int, string, bool, enum), use direct assignment
        else:
            setattr(target, field_name, value)


def _deep_merge_defaults(target: config_pb2.DefaultsConfig, source: config_pb2.DefaultsConfig) -> None:
    """Deep merge source defaults into target, field by field.

    Args:
        target: DefaultsConfig to merge into (modified in place)
        source: DefaultsConfig to merge from
    """
    if source.HasField("timeouts"):
        _merge_proto_fields(target.timeouts, source.timeouts)
    if source.HasField("ssh"):
        _merge_proto_fields(target.ssh, source.ssh)
    if source.HasField("autoscaler"):
        _merge_proto_fields(target.autoscaler, source.autoscaler)
    if source.HasField("bootstrap"):
        # Use standard merge for bootstrap fields, trusting HasField
        _merge_proto_fields(target.bootstrap, source.bootstrap)
        # Merge env_vars map separately (map fields don't use HasField)
        for key, value in source.bootstrap.env_vars.items():
            target.bootstrap.env_vars[key] = value


def _validate_autoscaler_config(config: config_pb2.AutoscalerConfig, context: str = "autoscaler") -> None:
    """Validate that autoscaler config has valid timing values.

    Assumes defaults have already been applied, so all fields must be set.
    If fields are missing, this will raise an error (as expected).

    Args:
        config: AutoscalerConfig to validate (with defaults applied)
        context: Description of where this config came from (for error messages)

    Raises:
        ValueError: If any timing value is invalid
    """
    # All fields must be set if defaults were applied
    interval_ms = config.evaluation_interval.milliseconds
    if interval_ms <= 0:
        raise ValueError(
            f"{context}: evaluation_interval must be positive, got {interval_ms}ms. "
            f"This controls how often the autoscaler evaluates scaling decisions."
        )

    timeout_ms = config.requesting_timeout.milliseconds
    if timeout_ms <= 0:
        raise ValueError(
            f"{context}: requesting_timeout must be positive, got {timeout_ms}ms. "
            f"This controls how long to wait for VMs to provision before timing out."
        )

    # scale_up_delay and scale_down_delay can be zero (no cooldown) but not negative
    if config.scale_up_delay.milliseconds < 0:
        raise ValueError(
            f"{context}: scale_up_delay must be non-negative, got {config.scale_up_delay.milliseconds}ms. "
            f"Use 0 for no cooldown after scaling up."
        )

    if config.scale_down_delay.milliseconds < 0:
        raise ValueError(
            f"{context}: scale_down_delay must be non-negative, got {config.scale_down_delay.milliseconds}ms. "
            f"Use 0 for no cooldown after scaling down."
        )


def apply_defaults(config: config_pb2.IrisClusterConfig) -> config_pb2.IrisClusterConfig:
    """Apply defaults to config and return merged result.

    Resolution order:
    1. Explicit field in config.defaults.* (if set)
    2. DEFAULT_CONFIG constant (hardcoded defaults)

    This function is called once during load_config().

    Args:
        config: Input cluster configuration

    Returns:
        New IrisClusterConfig with defaults fully resolved
    """
    merged = config_pb2.IrisClusterConfig()
    merged.CopyFrom(config)

    # Start with DEFAULT_CONFIG, then overlay user-provided defaults
    result_defaults = config_pb2.DefaultsConfig()
    result_defaults.CopyFrom(DEFAULT_CONFIG)

    # Merge each section
    if config.HasField("defaults"):
        _deep_merge_defaults(result_defaults, config.defaults)

    merged.defaults.CopyFrom(result_defaults)

    # Apply scale group defaults
    for group in merged.scale_groups.values():
        if not group.HasField("priority"):
            group.priority = 100

    # Validate merged autoscaler config
    if merged.defaults.HasField("autoscaler"):
        _validate_autoscaler_config(merged.defaults.autoscaler, context="config.defaults.autoscaler")

    return merged


def make_local_config(
    base_config: config_pb2.IrisClusterConfig,
) -> config_pb2.IrisClusterConfig:
    """Transform a GCP/manual config for local testing mode.

    This helper transforms any config to run locally:
    - Sets platform to local
    - Sets controller to local (in-process)
    - Sets all scale groups to local VMs
    - Applies LOCAL_DEFAULT_CONFIG for fast testing timings

    Args:
        base_config: Base configuration (should already have apply_defaults() called)

    Returns:
        Transformed config ready for local testing
    """
    config = config_pb2.IrisClusterConfig()
    config.CopyFrom(base_config)

    # Transform platform to local
    config.platform.ClearField("platform")
    config.platform.local.SetInParent()

    # Transform controller to local
    config.controller.ClearField("controller")
    config.controller.local.port = 0  # auto-assign
    config.controller.bundle_prefix = ""  # LocalController will set temp path

    # Transform all scale groups to local VMs
    for sg in config.scale_groups.values():
        sg.vm_type = config_pb2.VM_TYPE_LOCAL_VM

    # Apply local defaults (fast timings for testing)
    # Unconditionally use fast timings for local mode - this overrides any production timings
    # from DEFAULT_CONFIG that may have been applied during load_config()
    if not config.HasField("defaults"):
        config.defaults.CopyFrom(config_pb2.DefaultsConfig())

    # Set fast autoscaler timings for local testing
    config.defaults.autoscaler.evaluation_interval.CopyFrom(Duration.from_seconds(0.5).to_proto())
    config.defaults.autoscaler.scale_up_delay.CopyFrom(Duration.from_seconds(1).to_proto())
    # Keep scale_down_delay at 5min (same as production)
    if not config.defaults.autoscaler.HasField("scale_down_delay"):
        config.defaults.autoscaler.scale_down_delay.CopyFrom(Duration.from_seconds(300).to_proto())

    return config


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

    _normalize_scale_group_resources(data)

    # Ensure scale_groups have their name field set (proto uses map key, but config field needs it)
    if "scale_groups" in data:
        for name, sg_data in data["scale_groups"].items():
            if sg_data is None:
                data["scale_groups"][name] = {"name": name}
                sg_data = data["scale_groups"][name]
            elif "name" not in sg_data:
                sg_data["name"] = name

    # Normalize platform/controller oneof sections when YAML uses null values.
    # PyYAML parses:
    #   platform:
    #     local:
    # as {"platform": {"local": None}}, which ParseDict ignores for oneof fields.
    for section_key, oneof_keys in (
        ("platform", ("gcp", "manual", "local")),
        ("controller", ("gcp", "manual", "local")),
    ):
        if section_key in data and isinstance(data[section_key], dict):
            for oneof_key in oneof_keys:
                if oneof_key in data[section_key] and data[section_key][oneof_key] is None:
                    data[section_key][oneof_key] = {}

    # Convert lowercase accelerator types to enum format
    _normalize_accelerator_types(data)
    _normalize_vm_types(data)

    config = ParseDict(data, config_pb2.IrisClusterConfig())
    config = apply_defaults(config)
    _validate_accelerator_types(config)
    _validate_vm_types(config)
    _validate_scale_group_resources(config)

    platform_kind = config.platform.WhichOneof("platform") if config.HasField("platform") else "unspecified"
    logger.info(
        "Config loaded: platform=%s, scale_groups=%s",
        platform_kind,
        list(config.scale_groups.keys()) if config.scale_groups else "(none)",
    )

    return config


def _normalize_scale_group_resources(data: dict) -> None:
    """Normalize scale_group resources from YAML into proto-friendly fields."""
    scale_groups = data.get("scale_groups")
    if not isinstance(scale_groups, dict):
        return

    for name, sg in scale_groups.items():
        if not isinstance(sg, dict):
            continue

        resources = sg.get("resources")
        if resources is None:
            continue
        if not isinstance(resources, dict):
            raise ValueError(f"scale_groups.{name}.resources must be a mapping")

        normalized: dict[str, int] = {}

        cpu = resources.get("cpu")
        if cpu is not None:
            normalized["cpu"] = int(cpu)

        memory = resources.get("memory_bytes", resources.get("memory", resources.get("ram")))
        if memory is not None:
            normalized["memory_bytes"] = _parse_memory_value(memory, f"scale_groups.{name}.resources.ram")

        disk = resources.get("disk_bytes", resources.get("disk"))
        if disk is not None:
            normalized["disk_bytes"] = _parse_memory_value(disk, f"scale_groups.{name}.resources.disk")

        gpu = resources.get("gpu_count", resources.get("gpu"))
        if gpu is not None:
            normalized["gpu_count"] = int(gpu)

        tpu = resources.get("tpu_chips", resources.get("tpu"))
        if tpu is not None:
            normalized["tpu_chips"] = int(tpu)

        sg["resources"] = normalized


def _parse_memory_value(value: object, field_name: str) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(parse_memory_string(value))
    raise ValueError(f"{field_name} must be an int or size string (got {type(value).__name__})")


def config_to_dict(config: config_pb2.IrisClusterConfig) -> dict:
    """Convert config to dict for YAML serialization."""
    return MessageToDict(config, preserving_proto_field_name=True)


def get_ssh_config(
    cluster_config: config_pb2.IrisClusterConfig,
    group_name: str | None = None,
) -> SshConfig:
    """Get SSH config by merging cluster defaults with per-group overrides.

    Uses cluster_config.defaults.ssh for base settings:
    - user: "root"
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

    ssh = cluster_config.defaults.ssh
    user = ssh.user or DEFAULT_CONFIG.ssh.user
    key_file = ssh.key_file or None
    connect_timeout = (
        Duration.from_proto(ssh.connect_timeout)
        if ssh.HasField("connect_timeout") and ssh.connect_timeout.milliseconds > 0
        else Duration.from_proto(DEFAULT_CONFIG.ssh.connect_timeout)
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
    return SshConfig(
        user=user,
        key_file=key_file,
        port=DEFAULT_SSH_PORT,
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


def create_autoscaler(
    platform,  # type: Platform (avoiding circular import)
    autoscaler_config: config_pb2.AutoscalerConfig,
    scale_groups: dict[str, config_pb2.ScaleGroupConfig],
    dry_run: bool = False,
):
    """Create autoscaler from platform and explicit config.

    Args:
        platform: Platform instance for creating VM managers
        autoscaler_config: Autoscaler settings (already resolved with defaults)
        scale_groups: Map of scale group name to config
        dry_run: If True, don't actually provision VMs

    Returns:
        Configured Autoscaler instance

    Raises:
        ValueError: If autoscaler_config has invalid timing values
    """
    from iris.cluster.vm.autoscaler import Autoscaler

    # Validate autoscaler config before using it
    _validate_autoscaler_config(autoscaler_config, context="create_autoscaler")
    _validate_scale_group_resources(_scale_groups_to_config(scale_groups))

    # Create shared infrastructure
    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)

    # Extract autoscaler settings from config
    scale_up_delay = Duration.from_proto(autoscaler_config.scale_up_delay)
    scale_down_delay = Duration.from_proto(autoscaler_config.scale_down_delay)

    # Create scale groups using provided platform
    scaling_groups: dict[str, ScalingGroup] = {}
    for name, group_config in scale_groups.items():
        vm_manager = platform.vm_manager(group_config, vm_factory=vm_factory, dry_run=dry_run)

        scaling_groups[name] = ScalingGroup(
            config=group_config,
            vm_manager=vm_manager,
            scale_up_cooldown=scale_up_delay,
            scale_down_cooldown=scale_down_delay,
        )
        logger.info("Created scale group %s", name)

    # Create autoscaler using from_config classmethod
    return Autoscaler.from_config(
        scale_groups=scaling_groups,
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

    _validate_scale_group_resources(_scale_groups_to_config({name: spec.config for name, spec in specs.items()}))

    # Use provided config or DEFAULT_CONFIG
    if autoscaler_config is None:
        autoscaler_config = DEFAULT_CONFIG.autoscaler

    scale_up_delay = Duration.from_proto(autoscaler_config.scale_up_delay)
    scale_down_delay = Duration.from_proto(autoscaler_config.scale_down_delay)

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

    return Autoscaler.from_config(
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
    resources: config_pb2.ScaleGroupResources | None = None,
    slice_size: int = 1,
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

    if resources is None:
        raise ValueError("manual autoscaler requires explicit resources")

    sg_config = config_pb2.ScaleGroupConfig(
        name="manual",
        vm_type=config_pb2.VM_TYPE_MANUAL_VM,
        min_slices=0,
        max_slices=len(hosts),
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        resources=resources,
        slice_size=slice_size,
        zones=["local"],
    )

    timeouts = config_pb2.TimeoutConfig()
    timeouts.CopyFrom(DEFAULT_CONFIG.timeouts)

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


def create_local_autoscaler(
    config: config_pb2.IrisClusterConfig,
    controller_address: str,
    threads: ThreadContainer | None = None,
):
    """Create Autoscaler with LocalVmManagers for all scale groups.

    Creates its own temp directories for worker cache and bundles.
    The temp directory is stored as autoscaler._temp_dir for cleanup.

    Args:
        config: Cluster configuration (with defaults already applied)
        controller_address: Address for workers to connect to
        threads: Optional thread container for testing

    Returns:
        Configured Autoscaler with local VM managers
    """
    import tempfile

    from iris.cluster.vm.autoscaler import Autoscaler
    from iris.cluster.vm.local_platform import LocalVmManager, PortAllocator

    # Create temp dirs for worker resources (autoscaler owns these)
    temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_autoscaler_")
    temp_path = Path(temp_dir.name)
    cache_path = temp_path / "cache"
    cache_path.mkdir()
    fake_bundle = temp_path / "bundle"
    fake_bundle.mkdir()
    (fake_bundle / "pyproject.toml").write_text("[project]\nname='local'\n")

    vm_registry = VmRegistry()
    shared_port_allocator = PortAllocator(port_range=(30000, 40000))

    scale_groups: dict[str, ScalingGroup] = {}
    for name, sg_config in config.scale_groups.items():
        manager = LocalVmManager(
            scale_group_config=sg_config,
            controller_address=controller_address,
            cache_path=cache_path,
            fake_bundle=fake_bundle,
            vm_registry=vm_registry,
            port_allocator=shared_port_allocator,
        )
        scale_groups[name] = ScalingGroup(
            config=sg_config,
            vm_manager=manager,
            scale_up_cooldown=Duration.from_proto(config.defaults.autoscaler.scale_up_delay),
            scale_down_cooldown=Duration.from_proto(config.defaults.autoscaler.scale_down_delay),
        )

    autoscaler = Autoscaler.from_config(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=config.defaults.autoscaler,
        threads=threads,
    )
    # Store temp_dir for cleanup (caller should clean up via autoscaler._temp_dir)
    autoscaler._temp_dir = temp_dir
    return autoscaler


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


class IrisConfig:
    """Lightweight wrapper for IrisClusterConfig proto with component factories.

    Provides clean interface for creating Platform, Autoscaler, and other
    components from configuration without scattering factory logic across CLI.

    The proto is processed with apply_defaults() on construction, ensuring all
    default values are populated.

    Example:
        config = IrisConfig.load("cluster.yaml")
        platform = config.platform()

        # Use tunnel for connection
        with platform.tunnel(controller_address) as url:
            client = IrisClient.remote(url)
    """

    def __init__(self, proto: config_pb2.IrisClusterConfig):
        """Create IrisConfig from proto.

        Args:
            proto: Cluster configuration proto (defaults will be applied)
        """
        self._proto = apply_defaults(proto)

    @classmethod
    def load(cls, config_path: Path | str) -> "IrisConfig":
        """Load IrisConfig from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            IrisConfig with defaults applied and validated
        """
        proto = load_config(config_path)
        return cls(proto)

    @property
    def proto(self) -> config_pb2.IrisClusterConfig:
        """Access underlying proto (read-only)."""
        return self._proto

    def platform(self):
        """Create Platform instance from config.

        Returns:
            Platform implementation (GCP, Manual, or Local)
        """
        from iris.cluster.vm.platform import create_platform

        return create_platform(
            platform_config=self._proto.platform,
            bootstrap_config=self._proto.defaults.bootstrap,
            timeout_config=self._proto.defaults.timeouts,
            ssh_config=self._proto.defaults.ssh,
        )

    def as_local(self) -> "IrisConfig":
        """Create local variant of this config.

        Returns:
            New IrisConfig configured for local testing
        """
        local_proto = make_local_config(self._proto)
        return IrisConfig(local_proto)

    def controller_address(self) -> str:
        """Get controller address from bootstrap config, if set.

        Returns:
            Controller address string, or empty string if not configured
        """
        # TODO: Derive controller address from controller.manual/local when unset.
        bootstrap = self._proto.defaults.bootstrap
        if bootstrap.HasField("controller_address"):
            return bootstrap.controller_address
        return ""
