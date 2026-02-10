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

"""VM lifecycle management and platform abstractions.

Key components:
- VmManagerProtocol: Factory for creating VM groups (one per scale group)
- TpuVmManager: Creates TPU VM groups via gcloud
- ManualVmManager: Manages pre-existing hosts
- VmGroupProtocol, TpuVmGroup, ManualVmGroup: VM group lifecycle management
- VmRegistry: Centralized VM tracking for worker lookup
- ManagedVm: Per-VM lifecycle thread with bootstrap logic
- Platform: Multi-platform VM orchestration layer
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


# Platform abstraction
from iris.cluster.vm.platform import (
    Platform,
    PlatformOps,
    create_platform,
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
    "BootstrapError",
    "DirectSshConnection",
    "GceSshConnection",
    "GcloudSshConnection",
    "ManagedVm",
    "ManualVmGroup",
    "ManualVmManager",
    "Platform",
    "PlatformOps",
    "PoolExhaustedError",
    "QuotaExceededError",
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
    "cleanup_iris_resources",
    "connection_available",
    "create_platform",
    "discover_controller_vm",
    "list_docker_containers",
    "list_iris_tpus",
    "run_streaming_with_retry",
    "wait_for_connection",
]
