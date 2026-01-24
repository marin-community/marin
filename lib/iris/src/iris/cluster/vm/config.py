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
- create_autoscaler_from_config: Create from IrisClusterConfig
- create_autoscaler_from_specs: Create from explicit ScaleGroupSpec list
- create_manual_autoscaler: Quick path for manual hosts without config file
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from iris.cluster.vm.gcp_tpu_platform import TpuVmManager
from iris.cluster.vm.managed_vm import SshConfig, TrackedVmFactory, VmRegistry
from iris.cluster.vm.manual_platform import ManualVmManager
from iris.cluster.vm.vm_platform import VmManagerProtocol
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.rpc import vm_pb2

logger = logging.getLogger(__name__)


@dataclass
class ControllerVmConfig:
    """Controller VM configuration for GCP-managed controllers.

    For GCP controllers, set enabled=True and the system will create a VM.
    For manual SSH bootstrap, set host to the IP/hostname of an existing machine.
    """

    enabled: bool = False
    image: str = ""
    machine_type: str = "n2-standard-4"
    boot_disk_size_gb: int = 50
    port: int = 10000
    host: str = ""  # IP/hostname for manual controller SSH bootstrap


@dataclass
class IrisClusterConfig:
    """Unified configuration for VmManager instantiation.

    Mirrors the Ray cluster config structure for familiarity.
    """

    # Provider settings
    provider_type: str = "manual"  # "tpu" or "manual"
    project_id: str | None = None
    region: str | None = None
    zone: str | None = None

    # Auth settings
    ssh_user: str = "root"
    ssh_private_key: str | None = None

    # Docker/worker settings
    docker_image: str = ""
    worker_port: int = 10001

    # Controller settings
    controller_address: str = ""
    controller_vm: ControllerVmConfig = field(default_factory=ControllerVmConfig)

    # For manual provider
    manual_hosts: list[str] = field(default_factory=list)

    # Scale groups (name -> config)
    scale_groups: dict[str, vm_pb2.ScaleGroupConfig] = field(default_factory=dict)

    # Timeouts
    boot_timeout_seconds: int = 300
    init_timeout_seconds: int = 600
    ssh_connect_timeout_seconds: int = 30
    ssh_poll_interval_seconds: int = 5

    # GCP label prefix
    label_prefix: str = "iris"

    def to_bootstrap_config(self) -> vm_pb2.BootstrapConfig:
        """Convert to BootstrapConfig proto."""
        return vm_pb2.BootstrapConfig(
            controller_address=self.controller_address,
            docker_image=self.docker_image,
            worker_port=self.worker_port,
        )

    def to_timeout_config(self) -> vm_pb2.TimeoutConfig:
        """Convert to TimeoutConfig proto."""
        return vm_pb2.TimeoutConfig(
            boot_timeout_seconds=self.boot_timeout_seconds,
            init_timeout_seconds=self.init_timeout_seconds,
            ssh_connect_timeout_seconds=self.ssh_connect_timeout_seconds,
            ssh_poll_interval_seconds=self.ssh_poll_interval_seconds,
        )

    def to_ssh_config(self) -> SshConfig:
        """Convert to SshConfig for manual SSH."""
        return SshConfig(
            user=self.ssh_user,
            key_file=self.ssh_private_key,
            connect_timeout=self.ssh_connect_timeout_seconds,
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


def load_config(config_path: Path | str) -> IrisClusterConfig:
    """Load VmManager configuration from YAML file.

    YAML structure mirrors Ray cluster config:

    ```yaml
    provider:
      type: tpu  # or "manual"
      project_id: my-project
      region: us-east1
      zone: us-east1-d

    auth:
      ssh_user: ray
      ssh_private_key: ~/.ssh/key.pem

    docker:
      image: gcr.io/project/iris-worker:v1
      worker_port: 10001

    controller:
      address: "10.0.0.1:10000"

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

    # Parse provider section
    provider = data.get("provider", {})
    provider_type = provider.get("type", "manual")
    project_id = provider.get("project_id")
    region = provider.get("region")
    zone = provider.get("zone")

    # Parse auth section
    auth = data.get("auth", {})
    ssh_user = auth.get("ssh_user", "root")
    ssh_private_key = auth.get("ssh_private_key")

    # Parse docker section
    docker = data.get("docker", {})
    docker_image = docker.get("image", "")
    worker_port = docker.get("worker_port", 10001)

    # Parse controller section (expand env vars like ${IRIS_CONTROLLER_ADDRESS})
    controller = data.get("controller", {})
    controller_address = os.path.expandvars(controller.get("address", ""))

    # Parse controller VM settings
    controller_vm_data = controller.get("vm", {})
    controller_vm = ControllerVmConfig(
        enabled=controller_vm_data.get("enabled", False),
        image=controller_vm_data.get("image", ""),
        machine_type=controller_vm_data.get("machine_type", "n2-standard-4"),
        boot_disk_size_gb=controller_vm_data.get("boot_disk_size_gb", 50),
        port=controller_vm_data.get("port", 10000),
        host=controller_vm_data.get("host", ""),
    )

    # Only warn about missing controller address if controller VM is not enabled
    if not controller_address and not controller_vm.enabled:
        logger.warning("No controller address configured - workers will fail to start")

    # Parse manual hosts
    manual_hosts = data.get("manual_hosts", [])

    # Parse timeouts
    timeouts = data.get("timeouts", {})
    boot_timeout = timeouts.get("boot_timeout_seconds", 300)
    init_timeout = timeouts.get("init_timeout_seconds", 600)
    ssh_timeout = timeouts.get("ssh_connect_timeout_seconds", 30)
    ssh_poll = timeouts.get("ssh_poll_interval_seconds", 5)

    # Parse scale groups
    scale_groups: dict[str, vm_pb2.ScaleGroupConfig] = {}
    for name, sg_data in data.get("scale_groups", {}).items():
        zones = sg_data.get("zones", [])
        if zone and not zones:
            zones = [zone]

        scale_groups[name] = vm_pb2.ScaleGroupConfig(
            name=name,
            accelerator_type=sg_data.get("accelerator_type", ""),
            runtime_version=sg_data.get("runtime_version", ""),
            min_slices=sg_data.get("min_slices", 0),
            max_slices=sg_data.get("max_slices", 10),
            zones=zones,
            preemptible=sg_data.get("preemptible", False),
            priority=sg_data.get("priority", 100),
        )
        logger.debug("Loaded scale group %s: accelerator=%s, zones=%s", name, sg_data.get("accelerator_type"), zones)

    logger.info(
        "Config loaded: provider=%s, scale_groups=%s",
        provider_type,
        list(scale_groups.keys()) if scale_groups else "(none)",
    )

    return IrisClusterConfig(
        provider_type=provider_type,
        project_id=project_id,
        region=region,
        zone=zone,
        ssh_user=ssh_user,
        ssh_private_key=ssh_private_key,
        docker_image=docker_image,
        worker_port=worker_port,
        controller_address=controller_address,
        controller_vm=controller_vm,
        manual_hosts=manual_hosts,
        scale_groups=scale_groups,
        boot_timeout_seconds=boot_timeout,
        init_timeout_seconds=init_timeout,
        ssh_connect_timeout_seconds=ssh_timeout,
        ssh_poll_interval_seconds=ssh_poll,
    )


def create_autoscaler_from_config(
    config: IrisClusterConfig,
    autoscaler_config=None,
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

    Args:
        config: Cluster configuration with scale groups
        autoscaler_config: Optional autoscaler configuration

    Returns:
        A fully configured Autoscaler ready for use

    Raises:
        ValueError: If a scale group has an unknown provider type
    """
    from iris.cluster.vm.autoscaler import Autoscaler, AutoscalerConfig

    logger.info("Creating Autoscaler with provider=%s", config.provider_type)

    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)

    scale_groups: dict[str, ScalingGroup] = {}

    for name, group_config in config.scale_groups.items():
        manager = _create_manager(
            provider=config.provider_type,
            config=group_config,
            bootstrap_config=config.to_bootstrap_config(),
            timeouts=config.to_timeout_config(),
            vm_factory=vm_factory,
            project_id=config.project_id,
            hosts=config.manual_hosts,
            ssh_config=config.to_ssh_config(),
            label_prefix=config.label_prefix,
            dry_run=dry_run,
        )

        scale_groups[name] = ScalingGroup(
            config=group_config,
            vm_manager=manager,
        )

        logger.info(
            "Created scale group %s with provider=%s",
            name,
            config.provider_type,
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
