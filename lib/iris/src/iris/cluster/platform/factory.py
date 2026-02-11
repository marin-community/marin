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

"""Factory for creating Platform instances from cluster configuration."""

from typing import cast

from iris.cluster.platform.base import Platform
from iris.cluster.platform.coreweave import CoreweavePlatform
from iris.cluster.platform.gcp import GcpPlatform
from iris.cluster.platform.local import LocalPlatform
from iris.cluster.platform.manual import ManualPlatform
from iris.rpc import config_pb2


def create_platform(
    platform_config: config_pb2.PlatformConfig,
    ssh_config: config_pb2.SshConfig | None = None,
) -> Platform:
    """Create a Platform instance from configuration.

    Args:
        platform_config: Platform type and provider-specific settings.
        ssh_config: SSH settings (used by GCP and manual platforms).

    Returns:
        Platform instance for the configured provider type.

    Raises:
        ValueError: If platform type is unspecified or unknown.
    """
    if not platform_config.HasField("platform"):
        raise ValueError("platform is required")

    which = platform_config.WhichOneof("platform")
    label_prefix = platform_config.label_prefix or "iris"

    if which == "gcp":
        if not platform_config.gcp.project_id:
            raise ValueError("platform.gcp.project_id is required")
        return cast(
            Platform,
            GcpPlatform(
                gcp_config=platform_config.gcp,
                label_prefix=label_prefix,
                ssh_config=ssh_config,
            ),
        )

    if which == "manual":
        return cast(
            Platform,
            ManualPlatform(
                label_prefix=label_prefix,
                ssh_config=ssh_config,
            ),
        )

    if which == "local":
        return cast(Platform, LocalPlatform(label_prefix=label_prefix))

    if which == "coreweave":
        return cast(
            Platform,
            CoreweavePlatform(
                config=platform_config.coreweave,
                label_prefix=label_prefix,
            ),
        )

    raise ValueError(f"Unknown platform: {which}")
