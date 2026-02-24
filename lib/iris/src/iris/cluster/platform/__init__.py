# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Platform abstraction layer for infrastructure providers.

Provides a unified interface for creating/managing workers and slices across
different cloud providers (GCP, CoreWeave) and local testing.
"""

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    CommandResult,
    Labels,
    Platform,
    PlatformError,
    PlatformUnavailableError,
    QuotaExhaustedError,
    RemoteWorkerHandle,
    ResourceNotFoundError,
    SliceHandle,
    SliceStatus,
    StandaloneWorkerHandle,
    WorkerStatus,
)
from iris.cluster.platform.coreweave import (
    CoreweavePlatform,
)
from iris.cluster.platform.factory import (
    create_platform,
)
from iris.cluster.platform.gcp import (
    GcpPlatform,
    GcpSliceHandle,
    GcpStandaloneWorkerHandle,
    GcpWorkerHandle,
)
from iris.cluster.platform.local import (
    LocalPlatform,
    LocalSliceHandle,
)
from iris.cluster.platform.manual import (
    ManualPlatform,
    ManualSliceHandle,
    ManualStandaloneWorkerHandle,
    ManualWorkerHandle,
)

__all__ = [
    "CloudSliceState",
    "CloudWorkerState",
    "CommandResult",
    "CoreweavePlatform",
    "GcpPlatform",
    "GcpSliceHandle",
    "GcpStandaloneWorkerHandle",
    "GcpWorkerHandle",
    "Labels",
    "LocalPlatform",
    "LocalSliceHandle",
    "ManualPlatform",
    "ManualSliceHandle",
    "ManualStandaloneWorkerHandle",
    "ManualWorkerHandle",
    "Platform",
    "PlatformError",
    "PlatformUnavailableError",
    "QuotaExhaustedError",
    "RemoteWorkerHandle",
    "ResourceNotFoundError",
    "SliceHandle",
    "SliceStatus",
    "StandaloneWorkerHandle",
    "WorkerStatus",
    "create_platform",
]
