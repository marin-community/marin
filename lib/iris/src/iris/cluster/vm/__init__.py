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
    FakePopen,
    GceSshConnection,
    GcloudSshConnection,
    GcloudSshConnectionFactory,
    InMemorySshConnection,
    SshConnection,
    SshConnectionFactory,
    check_health,
    connection_available,
    make_direct_ssh_factory,
    make_gcloud_ssh_factory,
    make_in_memory_connection_factory,
    run_streaming_with_retry,
    shutdown_worker,
    wait_for_connection,
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
    IrisClusterConfig,
    ScaleGroupSpec,
    create_autoscaler_from_config,
    create_autoscaler_from_specs,
    create_manual_autoscaler,
    load_config,
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
    "FakePopen",
    "GceSshConnection",
    "GcloudSshConnection",
    "GcloudSshConnectionFactory",
    "GroupAvailability",
    "InMemorySshConnection",
    "IrisClusterConfig",
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
    "SshConnectionFactory",
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
    "connection_available",
    "create_autoscaler_from_config",
    "create_autoscaler_from_specs",
    "create_manual_autoscaler",
    "load_config",
    "make_direct_ssh_factory",
    "make_gcloud_ssh_factory",
    "make_in_memory_connection_factory",
    "route_demand",
    "run_streaming_with_retry",
    "shutdown_worker",
    "wait_for_connection",
]
