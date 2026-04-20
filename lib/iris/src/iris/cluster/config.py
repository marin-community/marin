# Copyright The Marin Authors
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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from google.protobuf.json_format import MessageToDict, ParseDict

from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.providers.k8s.tasks import K8sTaskProvider
from iris.cluster.providers.protocols import WorkerInfraProvider
from iris.cluster.controller.worker_provider import WorkerProvider
from iris.cluster.types import TPU_FAMILY_VARIANT_PREFIX, get_tpu_topology, parse_memory_string, tpu_variant_name
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import config_pb2
from iris.time_proto import duration_from_proto, duration_to_proto
from rigging.timing import Duration

logger = logging.getLogger(__name__)

DEFAULT_SSH_PORT = 22

# Re-export IrisClusterConfig from proto for public API
IrisClusterConfig = config_pb2.IrisClusterConfig

# Single source of truth for all default values
DEFAULT_CONFIG = config_pb2.DefaultsConfig(
    ssh=config_pb2.SshConfig(
        user="root",
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_METADATA,
        connect_timeout=duration_to_proto(Duration.from_seconds(30)),
    ),
    autoscaler=config_pb2.AutoscalerConfig(
        evaluation_interval=duration_to_proto(Duration.from_seconds(10)),
        scale_up_delay=duration_to_proto(Duration.from_seconds(60)),
        scale_down_delay=duration_to_proto(Duration.from_seconds(600)),
    ),
    worker=config_pb2.WorkerConfig(
        port=10001,
        cache_dir="/dev/shm/iris",
        host="0.0.0.0",
        port_range="30000-40000",
    ),
)

# Mapping from lowercase accelerator types to proto enum names
_ACCELERATOR_TYPE_MAP = {
    "cpu": "ACCELERATOR_TYPE_CPU",
    "gpu": "ACCELERATOR_TYPE_GPU",
    "tpu": "ACCELERATOR_TYPE_TPU",
}

_CAPACITY_TYPE_MAP = {
    "preemptible": "CAPACITY_TYPE_PREEMPTIBLE",
    "on_demand": "CAPACITY_TYPE_ON_DEMAND",
    "on-demand": "CAPACITY_TYPE_ON_DEMAND",
    "reserved": "CAPACITY_TYPE_RESERVED",
}

# Reverse mapping for YAML serialization: proto enum name → friendly YAML name
_CAPACITY_TYPE_REVERSE_MAP = {
    "CAPACITY_TYPE_PREEMPTIBLE": "preemptible",
    "CAPACITY_TYPE_ON_DEMAND": "on-demand",
    "CAPACITY_TYPE_RESERVED": "reserved",
}

_COREWEAVE_TOPOLOGY_LABEL_PREFIXES = (
    "backend.coreweave.cloud/",
    "ib.coreweave.cloud/",
    "node.coreweave.cloud/",
)


def _normalize_accelerator_type_field(d: dict) -> None:
    """Normalize a single accelerator_type field from lowercase to proto enum format."""
    accel_type = d.get("accelerator_type")
    if isinstance(accel_type, str):
        lower_type = accel_type.lower()
        if lower_type in _ACCELERATOR_TYPE_MAP:
            d["accelerator_type"] = _ACCELERATOR_TYPE_MAP[lower_type]


def _normalize_accelerator_types(data: dict) -> None:
    """Convert lowercase accelerator_type values to proto enum format on slice_templates.

    SliceConfig still carries accelerator_type for platform API use; the config
    loader derives it from resources but we must also normalize any explicit
    values that survive in slice_template (e.g. for local/demo configs).
    """
    if "scale_groups" not in data:
        return

    for sg_data in data["scale_groups"].values():
        if not sg_data:
            continue
        st = sg_data.get("slice_template")
        if st:
            _normalize_accelerator_type_field(st)


def _validate_accelerator_types(config: config_pb2.IrisClusterConfig) -> None:
    """Validate that scale groups have explicit device types in resources."""
    for name, sg_config in config.scale_groups.items():
        if not sg_config.HasField("resources"):
            continue
        if sg_config.resources.device_type == config_pb2.ACCELERATOR_TYPE_UNSPECIFIED:
            raise ValueError(f"Scale group '{name}' must set resources.device_type to cpu, gpu, or tpu.")


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
        if resources.cpu_millicores < 0:
            raise ValueError(f"Scale group '{name}' has invalid cpu_millicores={resources.cpu_millicores}.")
        if resources.memory_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid memory_bytes={resources.memory_bytes}.")
        if resources.disk_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid disk_bytes={resources.disk_bytes}.")
        if resources.device_count < 0:
            raise ValueError(f"Scale group '{name}' has invalid device_count={resources.device_count}.")
        if resources.capacity_type == config_pb2.CAPACITY_TYPE_UNSPECIFIED:
            raise ValueError(
                f"Scale group '{name}': resources.capacity_type is required "
                "(one of: preemptible, on-demand, reserved)."
            )


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
            resources = sg_config.resources
            gcp_mode = template.gcp.mode
            if gcp_mode == config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM:
                if resources.capacity_type != config_pb2.CAPACITY_TYPE_ON_DEMAND:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices only support capacity_type on-demand.")
                if sg_config.num_vms != 1:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices require num_vms=1.")
                if resources.device_type != config_pb2.ACCELERATOR_TYPE_CPU:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices currently require device_type=cpu.")
                if resources.device_variant:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices do not support device_variant.")
                if not template.gcp.machine_type:
                    raise ValueError(
                        f"Scale group '{name}': slice_template.gcp.machine_type must be non-empty for VM mode."
                    )
                if resources.device_count > 0:
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
    """Validate optional per-scale-group worker settings.

    Well-known attributes (device-type, device-variant, preemptible) are now
    auto-derived from resources, so we reject them in worker.attributes to
    prevent conflicting declarations.
    """
    _well_known_resource_attrs = {
        WellKnownAttribute.PREEMPTIBLE,
        WellKnownAttribute.DEVICE_TYPE,
        WellKnownAttribute.DEVICE_VARIANT,
        WellKnownAttribute.REGION,
        WellKnownAttribute.ZONE,
    }
    for name, sg_config in config.scale_groups.items():
        if not sg_config.HasField("worker"):
            continue

        attributes = sg_config.worker.attributes

        # Reject well-known keys that are derived elsewhere: device-type /
        # device-variant / preemptible come from `resources`; region / zone
        # come from `slice_template` (gcp.zone or coreweave.region).
        for attr_key in _well_known_resource_attrs:
            if attr_key in attributes:
                raise ValueError(
                    f"Scale group '{name}': worker.attributes.{attr_key} is derived automatically "
                    f"(from resources or slice_template) and must not be set explicitly. "
                    f"Remove it from worker.attributes."
                )

        template = sg_config.slice_template
        if (
            template.HasField("coreweave")
            and sg_config.resources.device_type == config_pb2.ACCELERATOR_TYPE_GPU
            and sg_config.num_vms > 1
        ):
            topology_attrs = {
                key: value
                for key, value in attributes.items()
                if any(key.startswith(prefix) for prefix in _COREWEAVE_TOPOLOGY_LABEL_PREFIXES)
            }
            if not topology_attrs:
                raise ValueError(
                    f"Scale group '{name}': CoreWeave GPU groups with num_vms>1 must set at least one "
                    "topology label in worker.attributes (for example "
                    "'backend.coreweave.cloud/superpod: same-slice')."
                )


def _derive_slice_config_from_resources(config: config_pb2.IrisClusterConfig) -> None:
    """Derive SliceConfig fields from ScaleGroupResources.

    Provider modules (gcp.py, local.py) read accelerator_type, accelerator_variant,
    capacity_type, and gpu_count from SliceConfig when calling cloud APIs. These fields
    are now the canonical source in resources; this function populates SliceConfig
    so provider modules continue to work without modification.

    Also derives disk_size_gb from resources.disk_bytes.
    """
    for sg_config in config.scale_groups.values():
        if not sg_config.HasField("resources") or not sg_config.HasField("slice_template"):
            continue

        resources = sg_config.resources
        template = sg_config.slice_template

        template.accelerator_type = resources.device_type
        if resources.device_variant:
            template.accelerator_variant = resources.device_variant
        template.capacity_type = resources.capacity_type

        if resources.device_type == config_pb2.ACCELERATOR_TYPE_GPU and resources.device_count > 0:
            template.gpu_count = resources.device_count

        if resources.disk_bytes:
            template.disk_size_gb = resources.disk_bytes // (1024**3)


def _validate_provider_platform_compat(config: config_pb2.IrisClusterConfig) -> None:
    """Reject unsupported provider + platform combinations.

    CoreweavePlatform no longer manages slices; configs that use
    ``platform.coreweave`` must use ``kubernetes_provider``.
    """
    is_coreweave = config.platform.WhichOneof("platform") == "coreweave"
    uses_worker_provider = config.WhichOneof("provider") == "worker_provider"
    if is_coreweave and uses_worker_provider:
        raise ValueError(
            "CoreWeave platform does not support worker_provider (CoreweavePlatform no longer "
            "manages slices). Use kubernetes_provider instead."
        )

    uses_k8s_provider = config.WhichOneof("provider") == "kubernetes_provider"
    if uses_k8s_provider and not config.kubernetes_provider.controller_address:
        raise ValueError(
            "kubernetes_provider.controller_address is required. Task pods need it to fetch "
            "bundles from the controller. Set it to the in-cluster service URL, e.g. "
            "http://iris-controller-svc.<namespace>.svc.cluster.local:<port>"
        )


def _validate_gcp_os_login_service_accounts(config: config_pb2.IrisClusterConfig) -> None:
    """Require explicit GCP service accounts when OS Login is enabled."""
    ssh_config = config.defaults.ssh
    if ssh_config.auth_mode != config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN:
        return

    controller_kind = config.controller.WhichOneof("controller")
    if controller_kind == "gcp" and not config.controller.gcp.service_account:
        raise ValueError(
            "controller.gcp.service_account is required when defaults.ssh.auth_mode=SSH_AUTH_MODE_OS_LOGIN."
        )

    for name, sg_config in config.scale_groups.items():
        template = sg_config.slice_template
        if template.WhichOneof("platform") != "gcp":
            continue
        if not template.gcp.service_account:
            raise ValueError(
                f"Scale group '{name}': slice_template.gcp.service_account is required when "
                "defaults.ssh.auth_mode=SSH_AUTH_MODE_OS_LOGIN."
            )


def validate_config(config: config_pb2.IrisClusterConfig) -> None:
    """Validate cluster config.

    Checks all scale groups for:
    - Required fields (name, resources, num_vms)
    - Device type is specified in resources
    - Resource values are non-negative
    - Slice templates have required platform-specific fields

    Raises:
        ValueError: If any validation constraint is violated
    """
    _validate_provider_platform_compat(config)
    _validate_accelerator_types(config)
    _validate_scale_group_resources(config)
    _validate_slice_templates(config)
    _validate_worker_settings(config)
    _validate_worker_defaults(config)
    _validate_gcp_os_login_service_accounts(config)


def _validate_worker_defaults(config: config_pb2.IrisClusterConfig) -> None:
    """Validate worker defaults required for worker-based platforms.

    Local platform runs workers in-process and does not require a docker image/runtime.
    GCP/manual/CoreWeave create remote worker processes and must provide a worker image.
    """
    # Some unit tests validate partial proto configs directly (without load_config/apply_defaults).
    # Only enforce worker image checks once defaults/platform are explicitly present.
    if not config.HasField("defaults"):
        return

    platform_kind = config.platform.WhichOneof("platform")
    if platform_kind in (None, "local"):
        return

    docker_image = config.defaults.worker.docker_image.strip()
    if not docker_image:
        raise ValueError("defaults.worker.docker_image is required for non-local platforms (gcp/manual/coreweave).")

    runtime = config.defaults.worker.runtime.strip()
    if runtime and runtime not in ("docker", "kubernetes"):
        raise ValueError(f"defaults.worker.runtime must be 'docker' or 'kubernetes', got {runtime!r}.")


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

        if field_desc.message_type is not None:
            target_field = getattr(target, field_name)
            target_field.CopyFrom(value)
        else:
            setattr(target, field_name, value)


def _deep_merge_defaults(target: config_pb2.DefaultsConfig, source: config_pb2.DefaultsConfig) -> None:
    """Deep merge source defaults into target, field by field.

    Sub-messages (timeouts, ssh, autoscaler, worker) are merged field-by-field
    so that partially-specified user configs overlay hardcoded defaults without
    wiping unset siblings. Top-level scalar fields are merged via
    _merge_proto_fields which copies any explicitly-set value.

    Args:
        target: DefaultsConfig to merge into (modified in place)
        source: DefaultsConfig to merge from
    """
    # Merge top-level scalar fields.
    # We skip message fields here since sub-messages need deep merging below.
    for field_desc in source.DESCRIPTOR.fields:
        if field_desc.message_type is not None:
            continue
        if source.HasField(field_desc.name):
            setattr(target, field_desc.name, getattr(source, field_desc.name))

    # Deep-merge sub-messages so partial overrides work
    if source.HasField("ssh"):
        _merge_proto_fields(target.ssh, source.ssh)
    if source.HasField("autoscaler"):
        _merge_proto_fields(target.autoscaler, source.autoscaler)
    if source.HasField("worker"):
        _merge_proto_fields(target.worker, source.worker)
        # Merge map fields separately (map fields don't support HasField)
        for key, value in source.worker.worker_attributes.items():
            target.worker.worker_attributes[key] = value
    # task_env is a top-level map on DefaultsConfig
    for key, value in source.task_env.items():
        target.task_env[key] = value


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

    # Populate the wire-format WorkerConfig.task_env from defaults.task_env.
    # Workers receive this via the autoscaler's base_worker_config.
    for key, value in merged.defaults.task_env.items():
        merged.defaults.worker.task_env[key] = value

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
    config.storage.remote_state_dir = ""  # LocalCluster will set temp path

    # Apply local defaults (fast timings for testing)
    # Unconditionally use fast timings for local mode - this overrides any production timings
    # from DEFAULT_CONFIG that may have been applied during load_config()
    if not config.HasField("defaults"):
        config.defaults.CopyFrom(config_pb2.DefaultsConfig())

    # Set fast autoscaler timings for local testing
    config.defaults.autoscaler.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.5)))
    config.defaults.autoscaler.scale_up_delay.CopyFrom(duration_to_proto(Duration.from_seconds(1)))
    # Use fast scale_down_delay for local dev (matching scale_up_delay)
    if not config.defaults.autoscaler.HasField("scale_down_delay"):
        config.defaults.autoscaler.scale_down_delay.CopyFrom(duration_to_proto(Duration.from_seconds(1)))

    return config


def _expand_tpu_pools(data: dict) -> None:
    """Expand ``tpu_pools`` into per-(size, zone) scale groups.

    Each pool defines shared properties for a TPU family. The ``sizes`` map
    lists per-size overrides (buffer_slices, max_slices, priority). For each
    pool x size x zone the function emits a fully-specified scale group with
    topology-derived fields (device_variant, num_vms, device_count) and
    autoscaler allocation metadata (quota_pool, allocation_tier, priority).

    Consumes the ``tpu_pools`` key from *data* and injects results into
    ``data["scale_groups"]``.
    """
    tpu_pools = data.pop("tpu_pools", None)
    if not tpu_pools:
        return
    if not isinstance(tpu_pools, dict):
        raise ValueError("tpu_pools must be a mapping")

    scale_groups = data.setdefault("scale_groups", {})

    for pool_name, pool in tpu_pools.items():
        if not isinstance(pool, dict):
            raise ValueError(f"tpu_pools.{pool_name} must be a mapping")

        family = pool.get("family")
        if not family or family not in TPU_FAMILY_VARIANT_PREFIX:
            raise ValueError(
                f"tpu_pools.{pool_name}: 'family' must be one of {sorted(TPU_FAMILY_VARIANT_PREFIX)}, got {family!r}"
            )

        zones = pool.get("zones")
        if not isinstance(zones, list) or not zones:
            raise ValueError(f"tpu_pools.{pool_name}: 'zones' must be a non-empty list")
        for z in zones:
            if not isinstance(z, str) or not z.strip():
                raise ValueError(f"tpu_pools.{pool_name}: each zone must be a non-empty string, got {z!r}")
        if len(zones) != len(set(zones)):
            raise ValueError(f"tpu_pools.{pool_name}: zones list contains duplicates: {zones}")

        sizes = pool.get("sizes")
        if not isinstance(sizes, dict) or not sizes:
            raise ValueError(f"tpu_pools.{pool_name}: 'sizes' must be a non-empty mapping")

        base_priority = pool.get("base_priority", 10)
        base_resources = pool.get("resources", {})
        base_slice_template = pool.get("slice_template", {})

        # Validate all sizes against the topology table up front
        sorted_sizes = sorted(sizes.keys(), key=lambda s: int(s))
        for size in sorted_sizes:
            variant = tpu_variant_name(family, int(size))
            try:
                get_tpu_topology(variant)
            except ValueError:
                raise ValueError(f"tpu_pools.{pool_name}.sizes.{size}: unknown TPU topology '{variant}'") from None

        for tier_index, size in enumerate(sorted_sizes):
            size_int = int(size)
            size_overrides = sizes[size] or {}
            variant = tpu_variant_name(family, size_int)
            topo = get_tpu_topology(variant)

            for zone in zones:
                sg_name = f"tpu_{pool_name}_{size_int}-{zone}"

                if sg_name in scale_groups:
                    raise ValueError(
                        f"tpu_pools.{pool_name}: expanded name '{sg_name}' collides with an existing scale group"
                    )

                # Build resources with topology-derived device fields
                resources = copy.deepcopy(base_resources)
                resources["device_type"] = "tpu"
                resources["device_variant"] = variant
                resources["device_count"] = topo.chips_per_vm

                # Build slice template with zone injected
                st = copy.deepcopy(base_slice_template)
                gcp = st.setdefault("gcp", {})
                gcp["zone"] = zone

                sg = {
                    "name": sg_name,
                    "num_vms": topo.vm_count,
                    "priority": size_overrides.get("priority", base_priority + tier_index * 10),
                    "quota_pool": f"{pool_name}/{zone}",
                    "allocation_tier": tier_index + 1,
                    "resources": resources,
                    "buffer_slices": size_overrides.get("buffer_slices", 0),
                    "max_slices": size_overrides["max_slices"],
                    "slice_template": st,
                }

                scale_groups[sg_name] = sg

    # Merge all TPU pool zones into platform.gcp.zones
    all_zones: set[str] = set()
    for pool in (tpu_pools or {}).values():
        if isinstance(pool, dict):
            for z in pool.get("zones", []):
                all_zones.add(z)
    if all_zones:
        platform = data.setdefault("platform", {})
        platform_gcp = platform.get("gcp")
        if isinstance(platform_gcp, dict):
            existing = set(platform_gcp.get("zones", []))
            existing.update(all_zones)
            platform_gcp["zones"] = sorted(existing)


def _expand_multi_zone_groups(data: dict) -> None:
    """Expand scale groups with `zones` into one group per zone.

    Consumes the YAML-only `zones` key on each scale group. For each zone,
    creates a copy of the scale group with:
    - name suffixed with -{zone} (e.g. tpu_v5e_16-europe-west4-b)
    - slice_template.gcp.zone set to the zone
    - buffer_slices defaulted to 0 if not explicitly set

    Also merges all expanded zones into platform.gcp.zones.

    Raises:
        ValueError: If zones is not a non-empty list of unique non-empty strings,
            if an expanded name collides with an existing scale group, or if
            slice_template.gcp.zone is set while zones is also specified.
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

        if existing_gcp_zone:
            raise ValueError(
                f"Scale group '{name}': cannot set both 'zones' and 'slice_template.gcp.zone'. "
                f"Remove slice_template.gcp.zone — it is set automatically by zone expansion."
            )

        for zone in zones:
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

            if "buffer_slices" not in expanded_sg:
                expanded_sg["buffer_slices"] = 0

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
    if "defaults" in data and "worker" in data["defaults"]:
        defaults_worker = data["defaults"]["worker"]
        if "controller_address" in defaults_worker:
            defaults_worker["controller_address"] = os.path.expandvars(defaults_worker["controller_address"])

    _expand_tpu_pools(data)
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
    _derive_slice_config_from_resources(config)
    validate_config(config)

    platform_kind = config.platform.WhichOneof("platform") if config.HasField("platform") else "unspecified"
    logger.debug(
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

    Also normalizes device_type from lowercase ("tpu") to proto enum format
    ("ACCELERATOR_TYPE_TPU") and converts device_count to the proto field.
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

        # Proto field names derived from the ScaleGroupResources descriptor,
        # plus fields handled explicitly by normalization code below (may not
        # yet appear in stale compiled protos).
        _NORMALIZED_KEYS = {"cpu", "ram", "disk", "capacity_type"}
        allowed_keys = set(config_pb2.ScaleGroupResources.DESCRIPTOR.fields_by_name.keys()) | _NORMALIZED_KEYS
        unknown_keys = set(resources.keys()) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"scale_groups.{name}.resources has unknown keys: {unknown}")

        normalized: dict[str, object] = {}

        cpu = resources.get("cpu")
        if cpu is not None:
            normalized["cpu_millicores"] = int(float(cpu) * 1000)
        elif "cpu_millicores" in resources:
            normalized["cpu_millicores"] = int(resources["cpu_millicores"])

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

        # Device configuration
        device_type = resources.get("device_type")
        if device_type is not None:
            if isinstance(device_type, str):
                lower = device_type.lower()
                if lower in _ACCELERATOR_TYPE_MAP:
                    normalized["device_type"] = _ACCELERATOR_TYPE_MAP[lower]
                else:
                    normalized["device_type"] = device_type
            else:
                normalized["device_type"] = device_type

        device_variant = resources.get("device_variant")
        if device_variant is not None:
            normalized["device_variant"] = str(device_variant)

        device_count = resources.get("device_count")
        if device_count is not None:
            normalized["device_count"] = int(device_count)

        capacity_type = resources.get("capacity_type")
        if capacity_type is not None:
            if isinstance(capacity_type, str):
                key = capacity_type.strip().lower().replace("-", "_")
                mapped = _CAPACITY_TYPE_MAP.get(key)
                if mapped is None:
                    allowed = ", ".join(sorted(_CAPACITY_TYPE_MAP.keys()))
                    raise ValueError(
                        f"scale_groups.{name}.resources.capacity_type must be one of "
                        f"{allowed}, got {capacity_type!r}"
                    )
                normalized["capacity_type"] = mapped
            elif isinstance(capacity_type, int):
                normalized["capacity_type"] = capacity_type
            else:
                raise ValueError(
                    f"scale_groups.{name}.resources.capacity_type must be a string, "
                    f"got {type(capacity_type).__name__}"
                )

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
            if "cpu_millicores" in resources:
                normalized["cpu"] = resources["cpu_millicores"] / 1000
            if "memory_bytes" in resources:
                normalized["ram"] = resources["memory_bytes"]
            if "disk_bytes" in resources:
                normalized["disk"] = resources["disk_bytes"]
            if "device_type" in resources:
                normalized["device_type"] = resources["device_type"]
            if "device_variant" in resources:
                normalized["device_variant"] = resources["device_variant"]
            if "device_count" in resources:
                normalized["device_count"] = resources["device_count"]
            if "capacity_type" in resources:
                raw_ct = resources["capacity_type"]
                normalized["capacity_type"] = _CAPACITY_TYPE_REVERSE_MAP.get(raw_ct, raw_ct)
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
    auth_mode = ssh.auth_mode if ssh.auth_mode else DEFAULT_CONFIG.ssh.auth_mode
    os_login_user = ssh.os_login_user or ""
    impersonate_service_account = ssh.impersonate_service_account or ""
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

    result = config_pb2.SshConfig(
        user=user,
        key_file=key_file,
        port=port,
        auth_mode=auth_mode,
        os_login_user=os_login_user,
        impersonate_service_account=impersonate_service_account,
    )
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

    Provides clean interface for creating provider bundles, autoscalers, and other
    components from configuration without scattering factory logic across CLI.

    The proto is processed with apply_defaults() on construction, ensuring all
    default values are populated.

    Example:
        config = IrisConfig.load("cluster.yaml")
        bundle = config.provider_bundle()

        # Use tunnel for connection
        with bundle.controller.tunnel(controller_address) as url:
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

    def workers(self):
        """Create WorkerInfraProvider instance from config.

        Returns:
            WorkerInfraProvider implementation (GCP, Manual, or Local).
            None for K8s/CoreWeave deployments.
        """
        return self.provider_bundle().workers

    def provider_bundle(self):
        """Create ControllerProvider + WorkerInfraProvider bundle from config.

        Returns:
            ProviderBundle with controller and optional workers
        """
        from iris.cluster.providers.factory import create_provider_bundle

        return create_provider_bundle(
            platform_config=self._proto.platform,
            cluster_config=self._proto,
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
        """Get controller address from worker config, if set.

        Returns:
            Controller address string, or empty string if not configured
        """
        # TODO: Derive controller address from controller.manual/local when unset.
        worker = self._proto.defaults.worker
        if worker.HasField("controller_address"):
            return worker.controller_address
        return ""


@contextmanager
def connect_cluster(config: config_pb2.IrisClusterConfig) -> Iterator[str]:
    """Start controller, open tunnel, yield address, stop on exit.

    Local mode uses LocalCluster directly (in-process controller + workers).
    Remote modes delegate controller lifecycle to the platform (GCP, CoreWeave, etc.).
    """
    validate_config(config)
    is_local = config.controller.WhichOneof("controller") == "local"

    if is_local:
        from iris.cluster.providers.local.cluster import LocalCluster

        cluster = LocalCluster(config)
        address = cluster.start()
        try:
            yield address
        finally:
            cluster.close()
    else:
        iris_config = IrisConfig(config)
        bundle = iris_config.provider_bundle()
        address = bundle.controller.start_controller(config)
        try:
            with bundle.controller.tunnel(address) as tunnel_url:
                yield tunnel_url
        finally:
            bundle.controller.stop_controller(config)
            bundle.controller.shutdown()


def create_autoscaler(
    platform: WorkerInfraProvider,
    autoscaler_config: config_pb2.AutoscalerConfig,
    scale_groups: dict[str, config_pb2.ScaleGroupConfig],
    label_prefix: str,
    base_worker_config: config_pb2.WorkerConfig | None = None,
    threads: ThreadContainer | None = None,
    db: "ControllerDB | None" = None,  # noqa: F821, UP037 — circular import
):
    """Create autoscaler from WorkerInfraProvider and explicit config.

    Args:
        platform: WorkerInfraProvider instance for creating/discovering slices
        autoscaler_config: Autoscaler settings (already resolved with defaults)
        scale_groups: Map of scale group name to config
        label_prefix: Prefix for labels on managed resources
        base_worker_config: Base worker configuration passed through to platform.create_slice().
            None disables bootstrap (test/local mode).
        threads: Thread container for background threads. Uses global default if not provided.
        db: Optional DB handle for write-through persistence.

    Returns:
        Configured Autoscaler instance

    Raises:
        ValueError: If autoscaler_config has invalid timing values
    """
    # Local import: controller modules import config.py, creating a circular dependency.
    from iris.cluster.controller.autoscaler import Autoscaler
    from iris.cluster.controller.autoscaler.scaling_group import (
        DEFAULT_SCALE_DOWN_RATE_LIMIT,
        DEFAULT_SCALE_UP_RATE_LIMIT,
        ScalingGroup,
    )

    threads = threads or get_thread_container()

    _validate_autoscaler_config(autoscaler_config, context="create_autoscaler")
    _validate_scale_group_resources(_scale_groups_to_config(scale_groups))

    scale_up_delay = duration_from_proto(autoscaler_config.scale_up_delay)
    scale_down_delay = duration_from_proto(autoscaler_config.scale_down_delay)

    scaling_groups: dict[str, ScalingGroup] = {}
    for name, group_config in scale_groups.items():
        scaling_groups[name] = ScalingGroup(
            config=group_config,
            platform=platform,
            label_prefix=label_prefix,
            scale_up_cooldown=scale_up_delay,
            idle_threshold=scale_down_delay,
            scale_up_rate_limit=group_config.scale_up_rate_limit or DEFAULT_SCALE_UP_RATE_LIMIT,
            scale_down_rate_limit=group_config.scale_down_rate_limit or DEFAULT_SCALE_DOWN_RATE_LIMIT,
            db=db,
        )
        resources = group_config.resources
        worker_attrs = dict(group_config.worker.attributes) if group_config.HasField("worker") else {}
        slice_template = group_config.slice_template
        cw_instance = slice_template.coreweave.instance_type if slice_template.HasField("coreweave") else ""
        logger.info(
            "Scale group %s: device=%s:%s device_count=%d num_vms=%d buffer=%d max=%d instance=%s worker_attrs=%s",
            name,
            resources.device_type,
            resources.device_variant,
            resources.device_count,
            group_config.num_vms,
            group_config.buffer_slices,
            group_config.max_slices,
            cw_instance or "n/a",
            worker_attrs or "none",
        )

    return Autoscaler.from_config(
        scale_groups=scaling_groups,
        config=autoscaler_config,
        platform=platform,
        base_worker_config=base_worker_config,
        db=db,
    )


def make_provider(cluster_config: config_pb2.IrisClusterConfig) -> WorkerProvider | K8sTaskProvider:
    """Create a TaskProvider from cluster configuration.

    Returns a K8sTaskProvider when `kubernetes_provider` is configured,
    or a WorkerProvider when `worker_provider` is configured.
    Raises ValueError if no provider is set.
    """
    which = cluster_config.WhichOneof("provider")
    if which == "kubernetes_provider":
        from iris.cluster.providers.k8s.service import CloudK8sService

        kp = cluster_config.kubernetes_provider
        namespace = kp.namespace or "iris"
        label_prefix = cluster_config.platform.label_prefix
        managed_label = f"iris-{label_prefix}-managed" if label_prefix else ""
        return K8sTaskProvider(
            kubectl=CloudK8sService(namespace=namespace, kubeconfig_path=kp.kubeconfig or None),
            namespace=namespace,
            default_image=kp.default_image,
            colocation_topology_key=kp.colocation_topology_key or "coreweave.cloud/spine",
            service_account=kp.service_account or "",
            host_network=kp.host_network,
            cache_dir=kp.cache_dir or "/cache",
            controller_address=kp.controller_address or None,
            managed_label=managed_label,
            task_env=dict(cluster_config.defaults.task_env),
        )
    if which == "worker_provider":
        from iris.cluster.controller.worker_provider import RpcWorkerStubFactory

        return WorkerProvider(stub_factory=RpcWorkerStubFactory())
    raise ValueError(
        "IrisClusterConfig.provider must be set. Add either:\n"
        "  worker_provider: {}\n"
        "or:\n"
        "  kubernetes_provider:\n"
        "    namespace: iris\n"
        "    default_image: ...\n"
        "to your cluster config."
    )


def clear_remote_state(remote_state_dir: str) -> None:
    """Remove all files under the remote state dir so the controller starts fresh."""
    import fsspec

    fs, path = fsspec.core.url_to_fs(remote_state_dir)
    if fs.exists(path):
        fs.rm(path, recursive=True)
