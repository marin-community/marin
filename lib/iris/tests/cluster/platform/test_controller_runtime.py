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

"""Tests for controller runtime wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from iris.cluster.platform.base import VmInfo, VmState
from iris.cluster.platform.controller_runtime import ControllerRuntime
from iris.rpc import config_pb2
from iris.time_utils import Timestamp


@dataclass
class _FakePlatform:
    list_result: list[VmInfo]
    start_result: list[VmInfo]
    started: int = 0
    stopped: list[str] | None = None

    def slice_manager(self, scale_group):
        raise NotImplementedError

    def tunnel(self, host, ports, *, zone=None):
        raise NotImplementedError

    def list_vms(self, *, tag=None, zone=None):
        return self.list_result

    def start_vms(self, spec, *, zone=None):
        self.started += 1
        return self.start_result

    def stop_vms(self, ids, *, zone=None):
        if self.stopped is None:
            self.stopped = []
        self.stopped.extend(ids)


def _manual_config() -> config_pb2.IrisClusterConfig:
    return config_pb2.IrisClusterConfig(
        platform=config_pb2.PlatformConfig(
            manual=config_pb2.ManualPlatformConfig(),
        ),
        controller=config_pb2.ControllerVmConfig(
            image="gcr.io/project/iris-controller:latest",
            manual=config_pb2.ManualControllerConfig(
                host="10.0.0.10",
                port=10000,
            ),
        ),
    )


def test_controller_runtime_start_uses_platform():
    vm = VmInfo(
        vm_id="controller-1",
        address="http://10.0.0.10:10000",
        zone=None,
        labels={},
        state=VmState.RUNNING,
        created_at_ms=Timestamp.now().epoch_ms(),
    )
    platform = _FakePlatform(list_result=[], start_result=[vm])
    runtime = ControllerRuntime(platform=platform, config=_manual_config())

    address = runtime.start()

    assert address == "http://10.0.0.10:10000"
    assert platform.started == 1


def test_controller_runtime_reuses_existing(monkeypatch: pytest.MonkeyPatch):
    vm = VmInfo(
        vm_id="controller-1",
        address="http://10.0.0.10:10000",
        zone=None,
        labels={},
        state=VmState.RUNNING,
        created_at_ms=Timestamp.now().epoch_ms(),
    )
    platform = _FakePlatform(list_result=[vm], start_result=[vm])
    runtime = ControllerRuntime(platform=platform, config=_manual_config())

    monkeypatch.setattr(
        "iris.cluster.platform.controller_runtime._check_health_rpc",
        lambda _address: True,
    )

    address = runtime.start()

    assert address == "http://10.0.0.10:10000"
    assert platform.started == 0


def test_controller_runtime_stop_uses_platform():
    vm = VmInfo(
        vm_id="controller-1",
        address="http://10.0.0.10:10000",
        zone=None,
        labels={},
        state=VmState.RUNNING,
        created_at_ms=Timestamp.now().epoch_ms(),
    )
    platform = _FakePlatform(list_result=[vm], start_result=[vm])
    runtime = ControllerRuntime(platform=platform, config=_manual_config())

    runtime.stop()

    assert platform.stopped == ["controller-1"]
