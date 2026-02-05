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

"""Controller lifecycle wrapper built on platform VM APIs."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.controller import (
    Controller as _InnerController,
    ControllerConfig as _InnerControllerConfig,
    RpcWorkerStubFactory,
)
from iris.cluster.platform import Platform
from iris.cluster.platform.base import ContainerSpec, VmBootstrapSpec, VmInfo
from iris.cluster.platform.bootstrap import (
    DEFAULT_BOOT_DISK_SIZE_GB,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_MACHINE_TYPE,
    build_controller_bootstrap_script,
)
from iris.cluster.platform.local import find_free_port
from iris.config import config_to_dict, create_local_autoscaler
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class ControllerStatus:
    """Status of the controller."""

    running: bool
    address: str | None
    healthy: bool
    vm_name: str | None = None


def _check_health_rpc(address: str, timeout: float = 5.0, log_result: bool = False) -> bool:
    """Check controller health via RPC."""
    try:
        client = ControllerServiceClientSync(address)
        client.list_jobs(cluster_pb2.Controller.ListJobsRequest(), timeout_ms=int(timeout * 1000))
        if log_result:
            logger.info("Health check %s: rpc_ok", address)
        return True
    except Exception as exc:
        if log_result:
            logger.info("Health check %s: rpc_error=%s", address, type(exc).__name__)
        return False


class _InProcessController(Protocol):
    """Protocol for the in-process Controller used by LocalController."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def url(self) -> str: ...


class LocalController:
    """In-process controller for local testing."""

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
        self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_controller_")
        bundle_dir = Path(self._temp_dir.name) / "bundles"
        bundle_dir.mkdir()

        port = self._config.controller.local.port or find_free_port()
        address = f"http://127.0.0.1:{port}"

        controller_threads = self._threads.create_child("controller") if self._threads else None
        autoscaler_threads = controller_threads.create_child("autoscaler") if controller_threads else None

        self._autoscaler = create_local_autoscaler(
            self._config,
            address,
            threads=autoscaler_threads,
        )

        self._controller = _InnerController(
            config=_InnerControllerConfig(
                host="127.0.0.1",
                port=port,
                bundle_prefix=self._config.controller.bundle_prefix or f"file://{bundle_dir}",
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
        if self._autoscaler and hasattr(self._autoscaler, "_temp_dir"):
            self._autoscaler._temp_dir.cleanup()
            self._autoscaler = None
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
        return ControllerStatus(running=False, address=None, healthy=False)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        return "(local controller — no startup logs)"


class ControllerVm:
    """Controller lifecycle wrapper for platform-managed VMs."""

    def __init__(
        self,
        platform: Platform,
        config: config_pb2.IrisClusterConfig,
        threads: ThreadContainer | None = None,
    ):
        self._platform = platform
        self._config = config
        self._threads = threads
        self._local_controller: LocalController | None = None
        self._last_vm: VmInfo | None = None
        self._label_prefix = config.platform.label_prefix or "iris"

    def start(self) -> str:
        if self._config.controller.WhichOneof("controller") == "local":
            return self._start_local()

        existing = self._discover()
        if existing and _check_health_rpc(existing.address):
            logger.info("Found existing controller at %s", existing.address)
            self._last_vm = existing
            return existing.address

        spec, zone = self._build_spec()
        vms = self._platform.start_vms(spec, zone=zone)
        if not vms:
            raise RuntimeError("Platform did not return a controller VM")
        self._last_vm = vms[0]
        return vms[0].address

    def stop(self) -> None:
        if self._config.controller.WhichOneof("controller") == "local":
            if self._local_controller:
                self._local_controller.stop()
                self._local_controller = None
            return

        ids: list[str] = []
        if self._last_vm:
            ids = [self._last_vm.vm_id]
        else:
            discovered = self._discover()
            if discovered:
                ids = [discovered.vm_id]
        if ids:
            _, zone = self._build_spec()
            self._platform.stop_vms(ids, zone=zone)
        self._last_vm = None

    def restart(self) -> str:
        self.stop()
        return self.start()

    def reload(self) -> str:
        return self.restart()

    def discover(self) -> str | None:
        if self._config.controller.WhichOneof("controller") == "local":
            if self._local_controller:
                return self._local_controller.discover()
            return None
        found = self._discover()
        return found.address if found else None

    def status(self) -> ControllerStatus:
        if self._config.controller.WhichOneof("controller") == "local":
            if self._local_controller:
                return self._local_controller.status()
            return ControllerStatus(running=False, address=None, healthy=False)

        found = self._discover()
        if not found:
            return ControllerStatus(running=False, address=None, healthy=False)
        healthy = _check_health_rpc(found.address)
        return ControllerStatus(running=True, address=found.address, healthy=healthy, vm_name=found.vm_id)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        return None

    def _start_local(self) -> str:
        if self._local_controller is None:
            self._local_controller = LocalController(self._config, threads=self._threads)
        return self._local_controller.start()

    def _discover(self) -> VmInfo | None:
        tag = f"{self._label_prefix}-controller"
        candidates = self._platform.list_vms(tag=tag)
        return candidates[0] if candidates else None

    def _build_spec(self) -> tuple[VmBootstrapSpec, str | None]:
        controller = self._config.controller
        image = controller.image
        if not image:
            raise RuntimeError("controller.image is required")

        port = DEFAULT_CONTROLLER_PORT
        provider = self._config.controller.WhichOneof("controller")
        if provider == "gcp":
            port = controller.gcp.port or port
        elif provider == "manual":
            port = controller.manual.port or port

        config_yaml = yaml.dump(config_to_dict(self._config), default_flow_style=False)
        bootstrap_script = build_controller_bootstrap_script(image, port, config_yaml)

        labels = {
            f"{self._label_prefix}-tag": f"{self._label_prefix}-controller",
            f"{self._label_prefix}-role": "controller",
        }

        container = ContainerSpec(
            image=image,
            entrypoint=[
                ".venv/bin/python",
                "-m",
                "iris.cluster.controller.main",
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            env={},
            ports={"http": port},
            health_port=port,
        )

        overrides: dict[str, object] = {
            "port": port,
            "vm_name": f"iris-controller-{self._label_prefix}",
        }

        zone: str | None = None
        if provider == "gcp":
            overrides.update(
                {
                    "machine_type": controller.gcp.machine_type or DEFAULT_MACHINE_TYPE,
                    "boot_disk_size_gb": controller.gcp.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB,
                }
            )
            if self._config.platform.gcp.zone:
                zone = self._config.platform.gcp.zone
            elif self._config.platform.gcp.default_zones:
                zone = self._config.platform.gcp.default_zones[0]
        elif provider == "manual":
            if not controller.manual.host:
                raise RuntimeError("controller.manual.host is required")
            overrides.update(
                {
                    "host": controller.manual.host,
                    "port": port,
                }
            )
        else:
            zone = None

        spec = VmBootstrapSpec(
            role="controller",
            container=container,
            labels=labels,
            bootstrap_script=bootstrap_script,
            provider_overrides=overrides,
        )
        return spec, zone
