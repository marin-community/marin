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

"""VM lifecycle management for the Iris autoscaler.

Key components:
- VmManagerProtocol: Factory for creating VM groups (one per scale group)
- TpuVmManager: Creates TPU VM groups via gcloud
- ManualVmManager: Manages pre-existing hosts
- VmGroupProtocol, TpuVmGroup, ManualVmGroup: VM group lifecycle management
- ScalingGroup: Owns VM groups for a scale group, tracks scaling state
- Autoscaler: Orchestrates multiple ScalingGroups
- VmRegistry: Centralized VM tracking for worker lookup
- ManagedVm: Per-VM lifecycle thread with bootstrap logic
"""

# SSH utilities
from iris.cluster.vm.ssh import (
    DirectSshConnection,
    GceSshConnection,
    GcloudSshConnection,
    SshConnection,
    connection_available,
    run_streaming_with_retry,
    wait_for_connection,
)

# Health checking (from controller module where diagnostics matter)
from iris.cluster.vm.controller import (
    HealthCheckResult,
    check_health,
)

# Platform protocols and status types
from iris.cluster.vm.vm_platform import (
    MAX_RECONCILE_WORKERS,
    VmGroupProtocol,
    VmGroupStatus,
    VmManagerProtocol,
    VmSnapshot,
)

# GCP TPU platform
from iris.cluster.vm.gcp_tpu_platform import (
    TpuVmGroup,
    TpuVmManager,
)

# Manual platform
from iris.cluster.vm.manual_platform import (
    ManualVmGroup,
    ManualVmManager,
)

# ManagedVm, registry, and factory
from iris.cluster.vm.managed_vm import (
    BOOTSTRAP_SCRIPT,
    PARTIAL_SLICE_GRACE_MS,
    BootstrapError,
    ManagedVm,
    PoolExhaustedError,
    QuotaExceededError,
    SshConfig,
    TrackedVmFactory,
    VmFactory,
    VmRegistry,
)

# Scaling group
from iris.cluster.vm.scaling_group import (
    AvailabilityState,
    GroupAvailability,
    ScalingGroup,
)

# Autoscaler
from iris.cluster.vm.autoscaler import (
    Autoscaler,
    AutoscalerConfig,
    DemandEntry,
    RoutingResult,
    ScalingAction,
    ScalingDecision,
    route_demand,
)

# Config and factory functions
from iris.cluster.vm.config import (
    ScaleGroupSpec,
    config_to_dict,
    create_autoscaler_from_config,
    create_autoscaler_from_specs,
    create_manual_autoscaler,
    get_ssh_config,
    load_config,
)

# Debug utilities
from iris.cluster.vm.debug import (
    cleanup_iris_resources,
    discover_controller_vm,
    list_docker_containers,
    list_iris_tpus,
)

__all__ = [
    "BOOTSTRAP_SCRIPT",
    "MAX_RECONCILE_WORKERS",
    "PARTIAL_SLICE_GRACE_MS",
    "Autoscaler",
    "AutoscalerConfig",
    "AvailabilityState",
    "BootstrapError",
    "DemandEntry",
    "DirectSshConnection",
    "GceSshConnection",
    "GcloudSshConnection",
    "GroupAvailability",
    "HealthCheckResult",
    "ManagedVm",
    "ManualVmGroup",
    "ManualVmManager",
    "PoolExhaustedError",
    "QuotaExceededError",
    "RoutingResult",
    "ScaleGroupSpec",
    "ScalingAction",
    "ScalingDecision",
    "ScalingGroup",
    "SshConfig",
    "SshConnection",
    "TpuVmGroup",
    "TpuVmManager",
    "TrackedVmFactory",
    "VmFactory",
    "VmGroupProtocol",
    "VmGroupStatus",
    "VmManagerProtocol",
    "VmRegistry",
    "VmSnapshot",
    "check_health",
    "cleanup_iris_resources",
    "config_to_dict",
    "connection_available",
    "create_autoscaler_from_config",
    "create_autoscaler_from_specs",
    "create_manual_autoscaler",
    "discover_controller_vm",
    "get_ssh_config",
    "list_docker_containers",
    "list_iris_tpus",
    "load_config",
    "route_demand",
    "run_streaming_with_retry",
    "wait_for_connection",
]
