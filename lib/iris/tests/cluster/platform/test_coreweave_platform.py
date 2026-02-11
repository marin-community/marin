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

"""Tests for CoreweavePlatform stub.

Verifies that the module imports cleanly and all Platform methods raise
NotImplementedError with a descriptive message.
"""

from __future__ import annotations

import pytest

from iris.cluster.platform.coreweave import CoreweavePlatform
from iris.rpc import config_pb2


@pytest.fixture
def platform() -> CoreweavePlatform:
    config = config_pb2.CoreweavePlatformConfig(region="us-east-1")
    return CoreweavePlatform(config, label_prefix="iris")


def test_create_vm_raises(platform: CoreweavePlatform) -> None:
    vm_config = config_pb2.VmConfig(name="test-vm")
    with pytest.raises(NotImplementedError, match="CoreWeave"):
        platform.create_vm(vm_config)


def test_create_slice_raises(platform: CoreweavePlatform) -> None:
    slice_config = config_pb2.SliceConfig(name_prefix="test")
    with pytest.raises(NotImplementedError, match="CoreWeave"):
        platform.create_slice(slice_config)


def test_list_slices_raises(platform: CoreweavePlatform) -> None:
    with pytest.raises(NotImplementedError, match="CoreWeave"):
        platform.list_slices(zones=["us-east-1a"])


def test_list_vms_raises(platform: CoreweavePlatform) -> None:
    with pytest.raises(NotImplementedError, match="CoreWeave"):
        platform.list_vms(zones=["us-east-1a"])


def test_tunnel_raises(platform: CoreweavePlatform) -> None:
    with pytest.raises(NotImplementedError, match="CoreWeave"):
        platform.tunnel("http://localhost:8080")


def test_shutdown_raises(platform: CoreweavePlatform) -> None:
    with pytest.raises(NotImplementedError, match="CoreWeave"):
        platform.shutdown()
