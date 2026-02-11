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

"""Local in-process controller for testing.

This module provides LocalController which runs the controller and autoscaler
in the current process for local testing. Workers are threads, not VMs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.config import create_local_autoscaler
from iris.cluster.controller.controller import (
    Controller as _InnerController,
    ControllerConfig as _InnerControllerConfig,
    RpcWorkerStubFactory,
)
from iris.cluster.vm.controller_vm import ControllerStatus
from iris.cluster.vm.local_platform import find_free_port
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_utils import Duration


class _InProcessController(Protocol):
    """Protocol for the in-process Controller used by LocalController.

    Avoids importing iris.cluster.controller.controller at module level
    which would create a circular dependency through the autoscaler.
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...

    @property
    def url(self) -> str: ...


class LocalController:
    """In-process controller for local testing.

    Runs Controller + Autoscaler(LocalVmManagers) in the current process.
    Workers are threads, not VMs. No Docker, no GCS, no SSH.
    """

    def __init__(
        self,
        config: config_pb2.IrisClusterConfig,
        threads: ThreadContainer | None = None,
    ):
        self._config = config
        self._threads = threads
        self._controller: _InProcessController | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._autoscaler: Autoscaler | None = None

    def start(self) -> str:
        # Create temp dir for controller's bundle storage
        self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_controller_")
        bundle_dir = Path(self._temp_dir.name) / "bundles"
        bundle_dir.mkdir()

        port = self._config.controller.local.port or find_free_port()
        address = f"http://127.0.0.1:{port}"

        controller_threads = self._threads.create_child("controller") if self._threads else None
        autoscaler_threads = controller_threads.create_child("autoscaler") if controller_threads else None

        # Autoscaler creates its own temp dirs for worker resources
        self._autoscaler = create_local_autoscaler(
            self._config,
            address,
            threads=autoscaler_threads,
        )

        worker_timeout = (
            Duration.from_proto(self._config.controller.worker_timeout)
            if self._config.controller.worker_timeout.milliseconds > 0
            else Duration.from_seconds(60.0)
        )
        self._controller = _InnerController(
            config=_InnerControllerConfig(
                host="127.0.0.1",
                port=port,
                bundle_prefix=self._config.controller.bundle_prefix or f"file://{bundle_dir}",
                worker_timeout=worker_timeout,
            ),
            worker_stub_factory=RpcWorkerStubFactory(),
            autoscaler=self._autoscaler,
            threads=controller_threads,
        )
        self._controller.start()
        return self._controller.url

    def stop(self) -> None:
        if self._controller:
            self._controller.stop()
            self._controller = None
        # Clean up autoscaler's temp dir
        if self._autoscaler and hasattr(self._autoscaler, "_temp_dir"):
            self._autoscaler._temp_dir.cleanup()
            self._autoscaler = None
        # Clean up controller's temp dir
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def restart(self) -> str:
        self.stop()
        return self.start()

    def reload(self) -> str:
        return self.restart()

    def discover(self) -> str | None:
        return self._controller.url if self._controller else None

    def status(self) -> ControllerStatus:
        if self._controller:
            return ControllerStatus(
                running=True,
                address=self._controller.url,
                healthy=True,
            )
        return ControllerStatus(running=False, address="", healthy=False)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        return "(local controller — no startup logs)"
