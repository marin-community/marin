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

"""Autoscaler factory functions for controller.

This module provides factory functions for creating autoscalers:
- create_autoscaler: Create from platform and explicit config
- create_autoscaler_from_specs: Create from explicit ScaleGroupSpec list
- create_manual_autoscaler: Quick path for manual hosts without config file
- create_local_autoscaler: Create for local testing with LocalVmManagers
"""

import logging
from pathlib import Path

from iris.cluster.config import DEFAULT_CONFIG, ScaleGroupSpec, _scale_groups_to_config, _validate_autoscaler_config
from iris.cluster.controller.scaling_group import DEFAULT_HEARTBEAT_GRACE, DEFAULT_STARTUP_GRACE, ScalingGroup
from iris.cluster.vm.gcp_tpu_platform import TpuVmManager
from iris.cluster.vm.managed_vm import SshConfig, TrackedVmFactory, VmRegistry
from iris.cluster.vm.manual_platform import ManualVmManager
from iris.cluster.vm.vm_platform import VmManagerProtocol
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


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
    from iris.cluster.controller.autoscaler import Autoscaler

    # Validate autoscaler config before using it
    _validate_autoscaler_config(autoscaler_config, context="create_autoscaler")
    from iris.cluster.config import _validate_scale_group_resources

    _validate_scale_group_resources(_scale_groups_to_config(scale_groups))

    # Create shared infrastructure
    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)

    # Extract autoscaler settings from config
    scale_up_delay = Duration.from_proto(autoscaler_config.scale_up_delay)
    scale_down_delay = Duration.from_proto(autoscaler_config.scale_down_delay)
    startup_grace = (
        Duration.from_proto(autoscaler_config.startup_grace_period)
        if autoscaler_config.startup_grace_period.milliseconds > 0
        else DEFAULT_STARTUP_GRACE
    )
    heartbeat_grace = (
        Duration.from_proto(autoscaler_config.heartbeat_grace_period)
        if autoscaler_config.heartbeat_grace_period.milliseconds > 0
        else DEFAULT_HEARTBEAT_GRACE
    )

    # Create scale groups using provided platform
    scaling_groups: dict[str, ScalingGroup] = {}
    for name, group_config in scale_groups.items():
        vm_manager = platform.vm_manager(group_config, vm_factory=vm_factory, dry_run=dry_run)

        scaling_groups[name] = ScalingGroup(
            config=group_config,
            vm_manager=vm_manager,
            scale_up_cooldown=scale_up_delay,
            scale_down_cooldown=scale_down_delay,
            startup_grace_period=startup_grace,
            heartbeat_grace_period=heartbeat_grace,
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
    from iris.cluster.controller.autoscaler import Autoscaler

    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)

    from iris.cluster.config import _validate_scale_group_resources

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

    from iris.cluster.controller.autoscaler import Autoscaler
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
