# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris cluster configuration loading and utilities.

Supports YAML config files for cluster management. This module provides:
- Configuration loading and validation (load_config, apply_defaults)
- Configuration serialization (config_to_dict)
- SSH configuration resolution (get_ssh_config)
- ScaleGroupSpec wrapper for extended group metadata
- IrisConfig high-level wrapper with component factories
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from google.protobuf.json_format import MessageToDict, ParseDict

from iris.cluster.types import PREEMPTIBLE_ATTRIBUTE_KEY, REGION_ATTRIBUTE_KEY, ZONE_ATTRIBUTE_KEY, parse_memory_string
from iris.managed_thread import ThreadContainer, get_thread_container
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


def _normalize_accelerator_type_field(d: dict) -> None:
    """Normalize a single accelerator_type field from lowercase to proto enum format."""
    accel_type = d.get("accelerator_type")
    if isinstance(accel_type, str):
        lower_type = accel_type.lower()
        if lower_type in _ACCELERATOR_TYPE_MAP:
            d["accelerator_type"] = _ACCELERATOR_TYPE_MAP[lower_type]


def _normalize_accelerator_types(data: dict) -> None:
    """Convert lowercase accelerator_type values to proto enum format.

    Modifies data in-place, converting values like "tpu" to "ACCELERATOR_TYPE_TPU"
    on both scale groups and their slice_templates.
    """
    if "scale_groups" not in data:
        return

    for sg_data in data["scale_groups"].values():
        if not sg_data:
            continue
        _normalize_accelerator_type_field(sg_data)

        st = sg_data.get("slice_template")
        if st:
            _normalize_accelerator_type_field(st)


def _validate_accelerator_types(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that scale groups have explicit accelerator types."""
    for name, sg_config in config.scale_groups.items():
        if sg_config.accelerator_type == config_pb2.ACCELERATOR_TYPE_UNSPECIFIED:
            raise ValueError(f"Scale group '{name}' must set accelerator_type to cpu, gpu, or tpu.")


def _validate_scale_group_resources(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that scale groups define per-VM resources and num_vms."""
    for name, sg_config in config.scale_groups.items():
        if not sg_config.HasField("resources"):
            raise ValueError(f"Scale group '{name}' must set resources.")
        if not sg_config.HasField("num_vms"):
            raise ValueError(f"Scale group '{name}' must set num_vms.")
        if sg_config.num_vms <= 0:
            raise ValueError(f"Scale group '{name}' has invalid num_vms={sg_config.num_vms}.")

        resources = sg_config.resources
        if resources.cpu < 0:
            raise ValueError(f"Scale group '{name}' has invalid cpu={resources.cpu}.")
        if resources.memory_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid memory_bytes={resources.memory_bytes}.")
        if resources.disk_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid disk_bytes={resources.disk_bytes}.")
        if resources.gpu_count < 0:
            raise ValueError(f"Scale group '{name}' has invalid gpu_count={resources.gpu_count}.")
        if resources.tpu_count < 0:
            raise ValueError(f"Scale group '{name}' has invalid tpu_count={resources.tpu_count}.")


def _validate_slice_templates(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that every scale group has a slice_template with a platform set.

    Each slice_template must declare a platform (gcp, manual, coreweave, local)
    with valid platform-specific fields.
    """
    for name, sg_config in config.scale_groups.items():
        if not sg_config.HasField("slice_template"):
            raise ValueError(f"Scale group '{name}': slice_template is required.")

        template = sg_config.slice_template
        platform = template.WhichOneof("platform")
        if platform is None:
            raise ValueError(
                f"Scale group '{name}': slice_template must have a platform (gcp, manual, coreweave, local)."
            )

        if platform == "gcp":
            if not template.gcp.zone:
                raise ValueError(f"Scale group '{name}': slice_template.gcp.zone must be non-empty.")
            gcp_mode = template.gcp.mode
            if gcp_mode == config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM:
                if template.preemptible:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices do not support preemptible instances.")
                if sg_config.num_vms != 1:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices require num_vms=1.")
                if sg_config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU:
                    raise ValueError(
                        f"Scale group '{name}': VM-backed GCP slices currently require accelerator_type=cpu."
                    )
                if template.accelerator_type not in (
                    config_pb2.ACCELERATOR_TYPE_UNSPECIFIED,
                    config_pb2.ACCELERATOR_TYPE_CPU,
                ):
                    raise ValueError(
                        f"Scale group '{name}': VM-backed GCP slices require slice_template.accelerator_type=cpu."
                    )
                if template.accelerator_variant:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices do not support accelerator_variant.")
                if not template.gcp.machine_type:
                    raise ValueError(
                        f"Scale group '{name}': slice_template.gcp.machine_type must be non-empty for VM mode."
                    )
                if sg_config.resources.tpu_count > 0 or sg_config.resources.gpu_count > 0:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices currently support CPU-only resources.")
            else:
                if not template.gcp.runtime_version:
                    raise ValueError(f"Scale group '{name}': slice_template.gcp.runtime_version must be non-empty.")
        elif platform == "manual":
            if not template.manual.hosts:
                raise ValueError(f"Scale group '{name}': slice_template.manual.hosts must be non-empty.")
        elif platform == "coreweave":
            if not template.coreweave.region:
                raise ValueError(f"Scale group '{name}': slice_template.coreweave.region must be non-empty.")
        elif platform == "local":
            pass

    if config.platform.HasField("gcp") and config.platform.gcp.zones:
        platform_zones = set(config.platform.gcp.zones)
        for name, sg_config in config.scale_groups.items():
            template = sg_config.slice_template
            if template.WhichOneof("platform") == "gcp" and template.gcp.zone:
                if template.gcp.zone not in platform_zones:
                    raise ValueError(
                        f"Scale group '{name}': zone '{template.gcp.zone}' is not in "
                        f"platform.gcp.zones {sorted(platform_zones)}. "
                        f"Add it to platform.gcp.zones."
                    )


def _validate_worker_settings(config: config_pb2.IrisClusterConfig) -> None:
    """Validate optional per-scale-group worker settings."""
    for name, sg_config in config.scale_groups.items():
        if not sg_config.HasField("worker"):
            continue

        attributes = sg_config.worker.attributes
        region = attributes.get(REGION_ATTRIBUTE_KEY, "").strip()
        if REGION_ATTRIBUTE_KEY in attributes and not region:
            raise ValueError(f"Scale group '{name}': worker.attributes.region must be non-empty.")

        preemptible_attr = attributes.get(PREEMPTIBLE_ATTRIBUTE_KEY, "")
        if PREEMPTIBLE_ATTRIBUTE_KEY in attributes:
            normalized = preemptible_attr.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(
                    f"Scale group '{name}': worker.attributes.preemptible must be 'true' or 'false',"
                    f" got {preemptible_attr!r}."
                )
            if normalized != str(bool(sg_config.slice_template.preemptible)).lower():
                raise ValueError(
                    f"Scale group '{name}': worker.attributes.preemptible={normalized!r} "
                    f"must match slice_template.preemptible={sg_config.slice_template.preemptible!r}."
                )

        zone_attr = attributes.get(ZONE_ATTRIBUTE_KEY, "").strip()
        if ZONE_ATTRIBUTE_KEY in attributes and not zone_attr:
            raise ValueError(f"Scale group '{name}': worker.attributes.zone must be non-empty.")
        if zone_attr and sg_config.slice_template.HasField("gcp") and sg_config.slice_template.gcp.zone:
            if zone_attr != sg_config.slice_template.gcp.zone:
                raise ValueError(
                    f"Scale group '{name}': worker.attributes.zone={zone_attr!r} must match "
                    f"slice_template.gcp.zone={sg_config.slice_template.gcp.zone!r}."
                )

        template = sg_config.slice_template
        if region and template.HasField("gcp") and template.gcp.zone:
            zone_region = template.gcp.zone.rsplit("-", 1)[0]
            if region != zone_region:
                raise ValueError(
                    f"Scale group '{name}': worker.attributes.region={region!r} must match "
                    f"slice_template.gcp.zone region {zone_region!r}."
                )


def validate_config(config: config_pb2.IrisClusterConfig) -> None:
    """Validate cluster config.

    Checks all scale groups for:
    - Required fields (name, resources, num_vms)
    - Enum fields are not UNSPECIFIED (accelerator_type)
    - Resource values are non-negative
    - Slice templates have required platform-specific fields

    Raises:
        ValueError: If any validation constraint is violated
    """
    _validate_accelerator_types(config)
    _validate_scale_group_resources(config)
    _validate_slice_templates(config)
    _validate_worker_settings(config)
    _validate_bootstrap_defaults(config)


def _validate_bootstrap_defaults(config: config_pb2.IrisClusterConfig) -> None:
    """Validate bootstrap defaults required for worker-based platforms.

    Local platform runs workers in-process and does not require bootstrap image/runtime.
    GCP/manual/CoreWeave create remote worker processes and must provide a worker image.
    """
    # Some unit tests validate partial proto configs directly (without load_config/apply_defaults).
    # Only enforce bootstrap image checks once defaults/platform are explicitly present.
    if not config.HasField("defaults"):
        return

    platform_kind = config.platform.WhichOneof("platform")
    if platform_kind in (None, "local"):
        return

    docker_image = config.defaults.bootstrap.docker_image.strip()
    if not docker_image:
        raise ValueError(
            "defaults.bootstrap.docker_image is required for non-local platforms " "(gcp/manual/coreweave)."
        )

    runtime = config.defaults.bootstrap.runtime.strip()
    if runtime and runtime not in {"docker", "kubernetes"}:
        raise ValueError(f"defaults.bootstrap.runtime must be one of docker/kubernetes, got {runtime!r}.")


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

    Sub-messages (timeouts, ssh, autoscaler, bootstrap) are merged field-by-field
    so that partially-specified user configs overlay hardcoded defaults without
    wiping unset siblings. Top-level scalar fields (e.g. default_task_image) are
    merged via _merge_proto_fields which copies any explicitly-set value.

    Args:
        target: DefaultsConfig to merge into (modified in place)
        source: DefaultsConfig to merge from
    """
    # Merge top-level scalar fields (e.g. default_task_image).
    # We skip message fields here since sub-messages need deep merging below.
    for field_desc in source.DESCRIPTOR.fields:
        if field_desc.message_type is not None:
            continue
        if source.HasField(field_desc.name):
            setattr(target, field_desc.name, getattr(source, field_desc.name))

    # Deep-merge sub-messages so partial overrides work
    if source.HasField("timeouts"):
        _merge_proto_fields(target.timeouts, source.timeouts)
    if source.HasField("ssh"):
        _merge_proto_fields(target.ssh, source.ssh)
    if source.HasField("autoscaler"):
        _merge_proto_fields(target.autoscaler, source.autoscaler)
    if source.HasField("bootstrap"):
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

    # Apply controller defaults
    if not merged.controller.HasField("heartbeat_failure_threshold"):
        merged.controller.heartbeat_failure_threshold = 10

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
    config.storage.bundle_prefix = ""  # LocalController will set temp path

    # Apply local defaults (fast timings for testing)
    # Unconditionally use fast timings for local mode - this overrides any production timings
    # from DEFAULT_CONFIG that may have been applied during load_config()
    if not config.HasField("defaults"):
        config.defaults.CopyFrom(config_pb2.DefaultsConfig())

    # Set fast controller timings for local testing.
    # worker_timeout is derived: heartbeat_interval * heartbeat_failure_threshold
    # = 0.5s * 3 = 1.5s effective timeout.
    config.controller.heartbeat_failure_threshold = 3

    # Set fast autoscaler timings for local testing
    config.defaults.autoscaler.evaluation_interval.CopyFrom(Duration.from_seconds(0.5).to_proto())
    config.defaults.autoscaler.scale_up_delay.CopyFrom(Duration.from_seconds(1).to_proto())
    # Keep scale_down_delay at 5min (same as production)
    if not config.defaults.autoscaler.HasField("scale_down_delay"):
        config.defaults.autoscaler.scale_down_delay.CopyFrom(Duration.from_seconds(300).to_proto())

    return config


def _expand_multi_zone_groups(data: dict) -> None:
    """Expand scale groups with `zones` into one group per zone.

    Consumes the YAML-only `zones` key on each scale group. For each zone,
    creates a copy of the scale group with:
    - name suffixed with -{zone} (e.g. tpu_v5e_16-europe-west4-b)
    - slice_template.gcp.zone set to the zone
    - worker.attributes.zone and worker.attributes.region set automatically
    - min_slices defaulted to 0 if not explicitly set

    Also merges all expanded zones into platform.gcp.zones.

    Raises:
        ValueError: If zones is not a non-empty list of unique non-empty strings,
            if an expanded name collides with an existing scale group, or if
            user-provided zone/region fields conflict with the expansion.
    """
    scale_groups = data.get("scale_groups")
    if not isinstance(scale_groups, dict):
        return

    all_expanded_zones: set[str] = set()
    expanded: dict[str, dict] = {}
    to_remove: list[str] = []

    for name, sg in list(scale_groups.items()):
        if not isinstance(sg, dict) or "zones" not in sg:
            continue

        zones = sg.pop("zones")
        if not isinstance(zones, list) or not zones:
            raise ValueError(f"Scale group '{name}': zones must be a non-empty list")

        for zone in zones:
            if not isinstance(zone, str) or not zone.strip():
                raise ValueError(f"Scale group '{name}': each zone must be a non-empty string, got {zone!r}")

        if len(zones) != len(set(zones)):
            raise ValueError(f"Scale group '{name}': zones list contains duplicates: {zones}")

        to_remove.append(name)

        # Zone expansion only makes sense for GCP slice templates.
        # If the template already specifies a non-GCP platform, reject it.
        st = sg.get("slice_template") or {}
        non_gcp_platforms = {"manual", "local", "coreweave"}
        specified_platforms = non_gcp_platforms & st.keys()
        if specified_platforms:
            raise ValueError(
                f"Scale group '{name}': 'zones' expansion is only supported for GCP slice templates, "
                f"but slice_template specifies {', '.join(sorted(specified_platforms))}."
            )

        # Detect conflicts with user-provided fields that expansion will set
        existing_gcp_zone = (sg.get("slice_template") or {}).get("gcp", {}).get("zone")
        existing_worker_attrs = (sg.get("worker") or {}).get("attributes", {})
        existing_zone_attr = existing_worker_attrs.get("zone")
        existing_region_attr = existing_worker_attrs.get("region")

        if existing_gcp_zone:
            raise ValueError(
                f"Scale group '{name}': cannot set both 'zones' and 'slice_template.gcp.zone'. "
                f"Remove slice_template.gcp.zone — it is set automatically by zone expansion."
            )
        if existing_zone_attr:
            raise ValueError(
                f"Scale group '{name}': cannot set both 'zones' and 'worker.attributes.zone'. "
                f"Remove worker.attributes.zone — it is set automatically by zone expansion."
            )
        if existing_region_attr:
            raise ValueError(
                f"Scale group '{name}': cannot set both 'zones' and 'worker.attributes.region'. "
                f"Remove worker.attributes.region — it is set automatically by zone expansion."
            )

        for zone in zones:
            region = zone.rsplit("-", 1)[0]
            expanded_name = f"{name}-{zone}"

            if expanded_name in scale_groups:
                raise ValueError(
                    f"Scale group '{name}': expanded name '{expanded_name}' collides with " f"an existing scale group."
                )
            if expanded_name in expanded:
                raise ValueError(
                    f"Scale group '{name}': expanded name '{expanded_name}' collides with " f"another expanded group."
                )

            expanded_sg = copy.deepcopy(sg)
            expanded_sg["name"] = expanded_name

            # Set zone in slice_template.gcp
            st = expanded_sg.setdefault("slice_template", {})
            gcp = st.setdefault("gcp", {})
            gcp["zone"] = zone

            # Set worker.attributes.zone and .region
            worker = expanded_sg.setdefault("worker", {})
            attrs = worker.setdefault("attributes", {})
            attrs[ZONE_ATTRIBUTE_KEY] = zone
            attrs[REGION_ATTRIBUTE_KEY] = region

            if "min_slices" not in expanded_sg:
                expanded_sg["min_slices"] = 0

            expanded[expanded_name] = expanded_sg
            all_expanded_zones.add(zone)

    for name in to_remove:
        del scale_groups[name]
    scale_groups.update(expanded)

    # Merge expanded zones into platform.gcp.zones
    if all_expanded_zones:
        platform = data.setdefault("platform", {})
        platform_gcp = platform.get("gcp")
        if isinstance(platform_gcp, dict):
            existing = set(platform_gcp.get("zones", []))
            existing.update(all_expanded_zones)
            platform_gcp["zones"] = sorted(existing)


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
    _expand_multi_zone_groups(data)

    # Ensure scale_groups have their name field set (proto uses map key, but config field needs it)
    if "scale_groups" in data:
        for name, sg_data in data["scale_groups"].items():
            if sg_data is None:
                data["scale_groups"][name] = {"name": name}
                sg_data = data["scale_groups"][name]
            elif "name" not in sg_data:
                sg_data["name"] = name

    # Normalize oneof sections when YAML uses null values.
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

    # Also normalize null oneof values inside slice_template blocks
    if "scale_groups" in data:
        for sg_data in data["scale_groups"].values():
            if not sg_data:
                continue
            st = sg_data.get("slice_template")
            if not st:
                continue
            for oneof_key in ("gcp", "manual", "local", "coreweave"):
                if oneof_key in st and st[oneof_key] is None:
                    st[oneof_key] = {}

    # Convert lowercase accelerator types to enum format
    _normalize_accelerator_types(data)

    config = ParseDict(data, config_pb2.IrisClusterConfig())
    config = apply_defaults(config)
    validate_config(config)

    platform_kind = config.platform.WhichOneof("platform") if config.HasField("platform") else "unspecified"
    logger.info(
        "Config loaded: platform=%s, scale_groups=%s",
        platform_kind,
        list(config.scale_groups.keys()) if config.scale_groups else "(none)",
    )

    return config


def _normalize_scale_group_resources(data: dict) -> None:
    """Normalize scale_group resources from YAML into proto-friendly fields.

    Accepts both YAML-friendly names (ram, disk) and proto field names
    (memory_bytes, disk_bytes) so configs serialized from protos (e.g.
    the controller ConfigMap JSON) can be loaded via load_config().
    """
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

        allowed_keys = {"cpu", "ram", "disk", "gpu_count", "tpu_count", "memory_bytes", "disk_bytes"}
        unknown_keys = set(resources.keys()) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"scale_groups.{name}.resources has unknown keys: {unknown}")

        normalized: dict[str, int] = {}

        cpu = resources.get("cpu")
        if cpu is not None:
            normalized["cpu"] = int(cpu)

        memory = resources.get("ram")
        if memory is not None:
            normalized["memory_bytes"] = _parse_memory_value(memory, f"scale_groups.{name}.resources.ram")
        elif "memory_bytes" in resources:
            normalized["memory_bytes"] = int(resources["memory_bytes"])

        disk = resources.get("disk")
        if disk is not None:
            normalized["disk_bytes"] = _parse_memory_value(disk, f"scale_groups.{name}.resources.disk")
        elif "disk_bytes" in resources:
            normalized["disk_bytes"] = int(resources["disk_bytes"])

        gpu = resources.get("gpu_count")
        if gpu is not None:
            normalized["gpu_count"] = int(gpu)

        tpu = resources.get("tpu_count")
        if tpu is not None:
            normalized["tpu_count"] = int(tpu)

        sg["resources"] = normalized


def _parse_memory_value(value: object, field_name: str) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(parse_memory_string(value))
    raise ValueError(f"{field_name} must be an int or size string (got {type(value).__name__})")


def config_to_dict(config: config_pb2.IrisClusterConfig) -> dict:
    """Convert config to dict for YAML serialization."""
    data = MessageToDict(config, preserving_proto_field_name=True)
    scale_groups = data.get("scale_groups")
    if isinstance(scale_groups, dict):
        for sg in scale_groups.values():
            if not isinstance(sg, dict):
                continue
            resources = sg.get("resources")
            if not isinstance(resources, dict):
                continue
            normalized: dict[str, object] = {}
            if "cpu" in resources:
                normalized["cpu"] = resources["cpu"]
            if "memory_bytes" in resources:
                normalized["ram"] = resources["memory_bytes"]
            if "disk_bytes" in resources:
                normalized["disk"] = resources["disk_bytes"]
            if "gpu_count" in resources:
                normalized["gpu_count"] = resources["gpu_count"]
            if "tpu_count" in resources:
                normalized["tpu_count"] = resources["tpu_count"]
            sg["resources"] = normalized
    return data


def get_ssh_config(
    cluster_config: config_pb2.IrisClusterConfig,
    group_name: str | None = None,
) -> config_pb2.SshConfig:
    """Get SSH config by merging cluster defaults with per-group overrides.

    Uses cluster_config.defaults.ssh for base settings:
    - user: "root"
    - port: 22
    - connect_timeout: 30s
    - key_file: "" (passwordless/agent auth)

    For manual providers, per-group overrides from scale_groups[*].manual take precedence.

    Args:
        cluster_config: The cluster configuration.
        group_name: Optional scale group name. If provided and the group uses
            manual VM type, per-group SSH overrides will be applied.

    Returns:
        config_pb2.SshConfig with all settings populated (using defaults where not specified).
    """
    ssh = cluster_config.defaults.ssh
    user = ssh.user or DEFAULT_CONFIG.ssh.user
    key_file = ssh.key_file or ""
    port = ssh.port if ssh.HasField("port") and ssh.port > 0 else DEFAULT_SSH_PORT
    connect_timeout = (
        ssh.connect_timeout
        if ssh.HasField("connect_timeout") and ssh.connect_timeout.milliseconds > 0
        else DEFAULT_CONFIG.ssh.connect_timeout
    )

    # Apply per-group overrides if group uses manual slice_template
    if group_name and group_name in cluster_config.scale_groups:
        group_config = cluster_config.scale_groups[group_name]
        if group_config.HasField("slice_template") and group_config.slice_template.HasField("manual"):
            manual = group_config.slice_template.manual
            if manual.ssh_user:
                user = manual.ssh_user
            if manual.ssh_key_file:
                key_file = manual.ssh_key_file

    result = config_pb2.SshConfig(user=user, key_file=key_file, port=port)
    result.connect_timeout.CopyFrom(connect_timeout)
    return result


@dataclass
class ScaleGroupSpec:
    """Extended scale group specification with provider info.

    Wraps a ScaleGroupConfig proto with additional metadata needed for
    factory instantiation, such as the provider type and manual hosts.
    """

    config: config_pb2.ScaleGroupConfig
    provider: str = "tpu"
    hosts: list[str] = field(default_factory=list)


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
    def load(cls, config_path: Path | str) -> IrisConfig:
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
        from iris.cluster.platform.factory import create_platform

        return create_platform(
            platform_config=self._proto.platform,
            ssh_config=self._proto.defaults.ssh,
        )

    def as_local(self) -> IrisConfig:
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


def create_autoscaler(
    platform,
    autoscaler_config: config_pb2.AutoscalerConfig,
    scale_groups: dict[str, config_pb2.ScaleGroupConfig],
    label_prefix: str,
    bootstrap_config: config_pb2.BootstrapConfig | None = None,
    threads: ThreadContainer | None = None,
):
    """Create autoscaler from Platform and explicit config.

    Args:
        platform: Platform instance for creating/discovering slices
        autoscaler_config: Autoscaler settings (already resolved with defaults)
        scale_groups: Map of scale group name to config
        label_prefix: Prefix for labels on managed resources
        bootstrap_config: Worker bootstrap settings passed through to platform.create_slice().
            None disables bootstrap (test/local mode).
        threads: Thread container for background threads. Uses global default if not provided.

    Returns:
        Configured Autoscaler instance

    Raises:
        ValueError: If autoscaler_config has invalid timing values
    """
    from iris.cluster.controller.autoscaler import Autoscaler
    from iris.cluster.controller.scaling_group import ScalingGroup

    threads = threads or get_thread_container()

    _validate_autoscaler_config(autoscaler_config, context="create_autoscaler")
    _validate_scale_group_resources(_scale_groups_to_config(scale_groups))

    scale_up_delay = Duration.from_proto(autoscaler_config.scale_up_delay)
    scale_down_delay = Duration.from_proto(autoscaler_config.scale_down_delay)

    scaling_groups: dict[str, ScalingGroup] = {}
    for name, group_config in scale_groups.items():
        scaling_groups[name] = ScalingGroup(
            config=group_config,
            platform=platform,
            label_prefix=label_prefix,
            scale_up_cooldown=scale_up_delay,
            scale_down_cooldown=scale_down_delay,
        )
        resources = group_config.resources
        worker_attrs = dict(group_config.worker.attributes) if group_config.HasField("worker") else {}
        slice_template = group_config.slice_template
        cw_instance = slice_template.coreweave.instance_type if slice_template.HasField("coreweave") else ""
        logger.info(
            "Scale group %s: accel=%s:%s gpu_count=%d min=%d max=%d instance=%s worker_attrs=%s",
            name,
            group_config.accelerator_type,
            group_config.accelerator_variant,
            resources.gpu_count,
            group_config.min_slices,
            group_config.max_slices,
            cw_instance or "n/a",
            worker_attrs or "none",
        )

    return Autoscaler.from_config(
        scale_groups=scaling_groups,
        config=autoscaler_config,
        platform=platform,
        bootstrap_config=bootstrap_config,
    )
