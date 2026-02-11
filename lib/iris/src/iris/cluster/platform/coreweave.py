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

"""CoreWeave platform stub.

Placeholder for future CoreWeave provider support. All methods raise
NotImplementedError until the implementation is built out.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from iris.cluster.platform.base import SliceHandle, StandaloneVmHandle, VmHandle
from iris.rpc import config_pb2


class CoreweavePlatform:
    """CoreWeave platform stub — not yet implemented.

    Implements the Platform interface shape but raises NotImplementedError
    for all methods. Wire this into create_platform() so that config
    validation catches CoreWeave usage early with a clear message.
    """

    def __init__(self, config: config_pb2.CoreweavePlatformConfig, label_prefix: str):
        self._config = config
        self._label_prefix = label_prefix

    def create_vm(self, config: config_pb2.VmConfig) -> StandaloneVmHandle:
        raise NotImplementedError("CoreWeave platform is not yet implemented")

    def create_slice(self, config: config_pb2.SliceConfig) -> SliceHandle:
        raise NotImplementedError("CoreWeave platform is not yet implemented")

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[SliceHandle]:
        raise NotImplementedError("CoreWeave platform is not yet implemented")

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[VmHandle]:
        raise NotImplementedError("CoreWeave platform is not yet implemented")

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        raise NotImplementedError("CoreWeave platform is not yet implemented")

    def shutdown(self) -> None:
        raise NotImplementedError("CoreWeave platform is not yet implemented")

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        raise NotImplementedError("CoreWeave platform is not yet implemented")
