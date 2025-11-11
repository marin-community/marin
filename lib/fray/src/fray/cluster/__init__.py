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

"""Fray cluster abstraction for job scheduling.

The cluster module provides a clean interface for launching and managing
jobs on different cluster backends. It supports both CLI-style job submissions
and can be used to configure distributed computation patterns.

Example:
    >>> from fray.cluster import LocalCluster, JobRequest, create_environment
    >>> cluster = LocalCluster()
    >>> request = JobRequest(
    ...     name="hello-world",
    ...     entrypoint="json.tool",
    ...     environment=create_environment(),
    ... )
    >>> job_id = cluster.launch(request)
    >>> for line in cluster.monitor(job_id):
    ...     print(line)
"""

from fray.cluster.base import Cluster
from fray.cluster.local import LocalCluster
from fray.cluster.types import (
    CpuConfig,
    DeviceConfig,
    EnvironmentConfig,
    GpuConfig,
    GpuType,
    JobId,
    JobInfo,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
    TpuType,
    create_environment,
)

__all__ = [
    "Cluster",
    "CpuConfig",
    "DeviceConfig",
    "EnvironmentConfig",
    "GpuConfig",
    "GpuType",
    "JobId",
    "JobInfo",
    "JobRequest",
    "JobStatus",
    "LocalCluster",
    "ResourceConfig",
    "TpuConfig",
    "TpuType",
    "create_environment",
]

# Ray cluster is optional
try:
    from fray.cluster.ray.cluster import RayCluster

    __all__.append("RayCluster")
except ImportError:
    pass
