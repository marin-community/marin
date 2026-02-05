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

"""Controller lifecycle wrapper built on generic platform VM APIs."""

from __future__ import annotations

import logging

import yaml

from iris.cluster.platform.base import ContainerSpec, VmBootstrapSpec, VmInfo
from iris.cluster.platform.controller_vm import (
    ControllerStatus,
    DEFAULT_BOOT_DISK_SIZE_GB,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_MACHINE_TYPE,
    LocalController,
    _build_controller_bootstrap_script,
    _check_health_rpc,
)
from iris.cluster.platform.platform import Platform
from iris.config import config_to_dict
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)


class ControllerRuntime:
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
        """Start controller, returning address (idempotent if healthy)."""
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
        """Stop controller VM if managed by this runtime."""
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
        """Stop then start controller."""
        self.stop()
        return self.start()

    def reload(self) -> str:
        """Reload controller by re-running bootstrap via restart for now."""
        return self.restart()

    def discover(self) -> str | None:
        """Find existing controller address, or None."""
        if self._config.controller.WhichOneof("controller") == "local":
            if self._local_controller:
                return self._local_controller.discover()
            return None
        found = self._discover()
        return found.address if found else None

    def status(self) -> ControllerStatus:
        """Get controller status."""
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
        """Fetch startup logs (unsupported for platform wrapper)."""
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
        bootstrap_script = _build_controller_bootstrap_script(image, port, config_yaml)

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
