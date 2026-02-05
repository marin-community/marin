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

"""Platform abstractions and factories for Iris."""

from __future__ import annotations

from iris.cluster.platform.base import Platform
from iris.rpc import config_pb2


def create_platform(
    platform_config: config_pb2.PlatformConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
    ssh_config: config_pb2.SshConfig,
) -> Platform:
    """Create platform from explicit config sections."""
    if not platform_config.HasField("platform"):
        raise ValueError("platform is required")

    which = platform_config.WhichOneof("platform")

    if which == "gcp":
        if not platform_config.gcp.project_id:
            raise ValueError("platform.gcp.project_id is required")
        from iris.cluster.platform.gcp import GcpPlatform

        return GcpPlatform(
            gcp_config=platform_config.gcp,
            label_prefix=platform_config.label_prefix or "iris",
            bootstrap_config=bootstrap_config,
            timeout_config=timeout_config,
        )

    if which == "manual":
        from iris.cluster.platform.manual import ManualPlatform

        return ManualPlatform(
            label_prefix=platform_config.label_prefix or "iris",
            bootstrap_config=bootstrap_config,
            timeout_config=timeout_config,
            ssh_config=ssh_config,
        )

    if which == "local":
        from iris.cluster.platform.local import LocalPlatform

        return LocalPlatform()

    raise ValueError(f"Unknown platform: {which}")


__all__ = [
    "Platform",
    "create_platform",
]
