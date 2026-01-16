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

"""Cluster client layer - "dumb" cluster operations with explicit parameters.

This package provides low-level cluster operations without context magic:
- ClusterOperations protocol: defines the interface for cluster operations
- RpcClusterOperations: RPC-based implementation (talks to controller)
- LocalClusterOperations: local thread-based implementation
- JobInfo: minimal job metadata (contextvar + environment variables)

For high-level operations with context magic, use fluster.client.
"""

from fluster.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from fluster.cluster.client.local import LocalClusterOperations
from fluster.cluster.client.protocols import ClusterOperations
from fluster.cluster.client.rpc import RpcClusterOperations

__all__ = [
    "ClusterOperations",
    "JobInfo",
    "LocalClusterOperations",
    "RpcClusterOperations",
    "get_job_info",
    "set_job_info",
]
