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
