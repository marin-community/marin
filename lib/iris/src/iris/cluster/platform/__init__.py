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

Provides a unified interface for creating/managing VMs and slices across
different cloud providers (GCP, CoreWeave) and local testing.
"""

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudVmState,
    CommandResult,
    Platform,
    PlatformError,
    PlatformUnavailableError,
    QuotaExhaustedError,
    ResourceNotFoundError,
    SliceHandle,
    SliceStatus,
    StandaloneVmHandle,
    VmHandle,
    VmStatus,
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
    GcpStandaloneVmHandle,
    GcpVmHandle,
)
from iris.cluster.platform.local import (
    LocalPlatform,
    LocalSliceHandle,
)
from iris.cluster.platform.manual import (
    ManualPlatform,
    ManualSliceHandle,
    ManualStandaloneVmHandle,
    ManualVmHandle,
)

__all__ = [
    "CloudSliceState",
    "CloudVmState",
    "CommandResult",
    "CoreweavePlatform",
    "GcpPlatform",
    "GcpSliceHandle",
    "GcpStandaloneVmHandle",
    "GcpVmHandle",
    "LocalPlatform",
    "LocalSliceHandle",
    "ManualPlatform",
    "ManualSliceHandle",
    "ManualStandaloneVmHandle",
    "ManualVmHandle",
    "Platform",
    "PlatformError",
    "PlatformUnavailableError",
    "QuotaExhaustedError",
    "ResourceNotFoundError",
    "SliceHandle",
    "SliceStatus",
    "StandaloneVmHandle",
    "VmHandle",
    "VmStatus",
    "create_platform",
]
