# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fray v1: legacy cluster and job APIs."""

from fray.v1.cluster import (
    Cluster,
    CpuConfig,
    Entrypoint,
    EnvironmentConfig,
    GpuConfig,
    JobId,
    JobInfo,
    JobRequest,
    LocalCluster,
    ResourceConfig,
    create_cluster,
    current_cluster,
)
from fray.v1.cluster.base import JobStatus
from fray.v1.isolated_env import JobGroup, TemporaryVenv

__all__ = [
    "Cluster",
    "CpuConfig",
    "Entrypoint",
    "EnvironmentConfig",
    "GpuConfig",
    "JobGroup",
    "JobId",
    "JobInfo",
    "JobRequest",
    "JobStatus",
    "LocalCluster",
    "ResourceConfig",
    "TemporaryVenv",
    "create_cluster",
    "current_cluster",
]
