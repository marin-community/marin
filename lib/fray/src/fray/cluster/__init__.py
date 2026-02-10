# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fray.cluster now re-exports v2 types.

For v1 cluster APIs (Cluster, current_cluster, create_cluster, etc.),
use ``fray.v1.cluster`` instead.
"""

from fray.v2.types import (
    CpuConfig,
    DeviceConfig,
    DeviceKind,
    Entrypoint,
    EnvironmentConfig,
    GpuConfig,
    GpuType,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
    TpuTopologyInfo,
    TpuType,
    create_environment,
    get_tpu_topology,
)

__all__ = [
    "CpuConfig",
    "DeviceConfig",
    "DeviceKind",
    "Entrypoint",
    "EnvironmentConfig",
    "GpuConfig",
    "GpuType",
    "JobRequest",
    "JobStatus",
    "ResourceConfig",
    "TpuConfig",
    "TpuTopologyInfo",
    "TpuType",
    "create_environment",
    "get_tpu_topology",
]
