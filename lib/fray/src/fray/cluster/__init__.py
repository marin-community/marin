# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fray.cluster re-exports resource/job types for back-compat."""

from fray.types import (
    ANY_REGION,
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
    "ANY_REGION",
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
