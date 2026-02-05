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

"""Cluster lifecycle manager.

Provides a uniform interface for starting/stopping/connecting to an Iris
cluster regardless of backend (GCP, manual, local). Callers get a URL;
ClusterManager handles tunnel setup, mode detection, and cleanup.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.controller import (
    Controller as _InnerController,
    ControllerConfig as _InnerControllerConfig,
    RpcWorkerStubFactory,
)
from iris.cluster.platform.base import ContainerSpec, Platform, VmBootstrapSpec
from iris.cluster.platform.bootstrap import (
    DEFAULT_BOOT_DISK_SIZE_GB,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_MACHINE_TYPE,
    build_controller_bootstrap_script,
)
from iris.cluster.platform.local import find_free_port
from iris.config import IrisConfig, config_to_dict, create_local_autoscaler
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ControllerStatus:
    """Status of the controller."""

    running: bool
    address: str | None
    healthy: bool
    vm_name: str | None = None


@dataclass(frozen=True)
class StopClusterResult:
    """Results from stopping a cluster and deleting slices."""

    discovered: dict[str, dict[str | None, list[str]]]
    failed: dict[str, dict[str | None, dict[str, str]]]


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


class _LocalController:
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

    def status(self) -> _ControllerStatus:
        if self._controller:
            return _ControllerStatus(
                running=True,
                address=self._controller.url,
                healthy=True,
            )
        return _ControllerStatus(running=False, address=None, healthy=False)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        del tail_lines
        return "(local controller — no startup logs)"


class _ControllerVm:
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
        self._local_controller: _LocalController | None = None
        self._last_vm: vm_pb2.VmInfo | None = None
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

    def status(self) -> _ControllerStatus:
        if self._config.controller.WhichOneof("controller") == "local":
            if self._local_controller:
                return self._local_controller.status()
            return _ControllerStatus(running=False, address=None, healthy=False)

        found = self._discover()
        if not found:
            return _ControllerStatus(running=False, address=None, healthy=False)
        healthy = _check_health_rpc(found.address)
        return _ControllerStatus(running=True, address=found.address, healthy=healthy, vm_name=found.vm_id)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        del tail_lines
        return None

    def _start_local(self) -> str:
        if self._local_controller is None:
            self._local_controller = _LocalController(self._config, threads=self._threads)
        return self._local_controller.start()

    def _discover(self) -> vm_pb2.VmInfo | None:
        tag = f"{self._label_prefix}-controller"
        candidates = self._platform.list_vms(tag=tag)
        return candidates[0] if candidates else None

    def _build_spec(self) -> tuple[VmBootstrapSpec, str | None]:
        controller = self._config.controller
        image = controller.image
        if not image:
            raise RuntimeError("controller.image is required")

        port = None
        machine_type = DEFAULT_MACHINE_TYPE
        disk_size = DEFAULT_BOOT_DISK_SIZE_GB
        zone: str | None = None

        if controller.HasField("gcp"):
            machine_type = controller.gcp.machine_type or DEFAULT_MACHINE_TYPE
            disk_size = controller.gcp.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB
            port = controller.gcp.port or DEFAULT_CONTROLLER_PORT
            zone = self._config.platform.gcp.zone or None

        if controller.HasField("manual"):
            port = controller.manual.port or DEFAULT_CONTROLLER_PORT

        if controller.HasField("local"):
            port = controller.local.port or DEFAULT_CONTROLLER_PORT

        config_dict = config_to_dict(self._config)
        config_yaml = yaml.safe_dump(config_dict)
        bootstrap_script = build_controller_bootstrap_script(
            docker_image=image,
            port=port or DEFAULT_CONTROLLER_PORT,
            config_yaml=config_yaml,
        )

        spec = VmBootstrapSpec(
            role="controller",
            container=ContainerSpec(
                image=image,
                entrypoint=[],
                env={},
                ports={"controller": port or DEFAULT_CONTROLLER_PORT},
                health_port=port or DEFAULT_CONTROLLER_PORT,
            ),
            labels={"iris-role": "controller"},
            bootstrap_script=bootstrap_script,
            provider_overrides={
                "machine_type": machine_type,
                "boot_disk_size_gb": disk_size,
                "metadata": {"iris-config": config_yaml},
            },
        )
        return spec, zone


class ClusterManager:
    """Manages the full cluster lifecycle: controller + connectivity.

    Provides explicit start/stop methods and a connect() context manager
    that handles tunnel setup for GCP or direct connection for local mode.

    Example (smoke test / demo):
        manager = ClusterManager(config)
        with manager.connect() as url:
            client = IrisClient.remote(url)
            client.submit(...)

    Example (CLI - long-running):
        manager = ClusterManager(config)
        url = manager.start()
        # ... use cluster ...
        manager.stop()
    """

    def __init__(
        self,
        config: config_pb2.IrisClusterConfig,
        threads: ThreadContainer | None = None,
        platform: Platform | None = None,
    ):
        self._config = config
        self._threads = threads if threads is not None else get_thread_container()
        self._iris_config = IrisConfig(config)
        self._platform = platform if platform is not None else self._iris_config.platform()
        self._controller: _ControllerVm | None = None

    @property
    def is_local(self) -> bool:
        return self._config.controller.WhichOneof("controller") == "local"

    def start(self) -> str:
        """Start the controller. Returns the controller address.

        For GCP: creates a GCE VM, bootstraps, returns internal IP.
        For local: starts in-process Controller, returns localhost URL.
        """
        controller = self._ensure_controller()
        address = controller.start()
        logger.info("Controller started at %s (local=%s)", address, self.is_local)
        return address

    def stop(self) -> None:
        """Stop the controller and clean up resources.

        Shutdown ordering:
        1. Stop the controller (which stops its threads and autoscaler)
        2. Wait on the root ThreadContainer to verify all threads have exited
        """
        self.stop_controller()
        self._threads.wait()

    def stop_controller(self) -> None:
        """Stop the controller if present (even if not started in this process)."""
        controller = self._ensure_controller()
        controller.stop()
        self._controller = None
        logger.info("Controller stopped")

    def stop_cluster(self, *, zone: str | None = None) -> StopClusterResult:
        """Stop controller and terminate all slices.

        Returns:
            StopClusterResult containing discovered slice IDs and failures.
        """
        discovered: dict[str, dict[str | None, list[str]]] = {}
        for name, group_config in self._config.scale_groups.items():
            for zone_name in self._resolve_group_zones(group_config, self._config.platform, zone):
                slices = self._platform.list_slices(group_config, zone=zone_name)
                slice_ids = [s.slice_id for s in slices]
                if slice_ids:
                    discovered.setdefault(name, {}).setdefault(zone_name, []).extend(slice_ids)

        self.stop_controller()

        failed: dict[str, dict[str | None, dict[str, str]]] = {}
        for name, per_zone in discovered.items():
            group_config = self._config.scale_groups[name]
            for zone_name, slice_ids in per_zone.items():
                for slice_id in slice_ids:
                    try:
                        self._platform.delete_slice(group_config, slice_id, zone=zone_name)
                    except Exception as exc:
                        failed.setdefault(name, {}).setdefault(zone_name, {})[slice_id] = str(exc)
        return StopClusterResult(discovered=discovered, failed=failed)

    def reload(self) -> str:
        """Reload cluster by reloading the controller.

        The controller will re-bootstrap all worker VMs when it starts,
        ensuring they run the latest worker image. This is faster than
        a full stop/start cycle and ensures consistent image versions.

        For GCP: Reloads controller VM, which then re-bootstraps workers.
        For local: Equivalent to restart (no VMs to preserve).

        Returns:
            Controller address

        Raises:
            RuntimeError: If controller VM doesn't exist
        """
        controller = self._ensure_controller()
        address = controller.reload()
        logger.info("Controller reloaded at %s", address)

        return address

    def discover_controller(self) -> str | None:
        """Return the controller address if discoverable (no start)."""
        controller = self._ensure_controller()
        return controller.discover()

    def controller_status(self) -> _ControllerStatus:
        """Return controller status (running/healthy/address)."""
        controller = self._ensure_controller()
        return controller.status()

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        """Fetch controller startup logs if available."""
        controller = self._ensure_controller()
        return controller.fetch_startup_logs(tail_lines=tail_lines)

    @contextmanager
    def connect(self) -> Iterator[str]:
        """Start controller, yield a usable URL, stop on exit.

        For GCP: establishes SSH tunnel, yields tunnel URL.
        For local: yields direct localhost URL (no tunnel).
        """
        address = self.start()
        try:
            # Use Platform.tunnel() for consistent connection handling
            with self._platform.tunnel(address) as tunnel_url:
                yield tunnel_url
        finally:
            self.stop()

    def _ensure_controller(self) -> _ControllerVm:
        if self._controller is None:
            self._controller = _ControllerVm(self._platform, self._iris_config.proto, threads=self._threads)
        return self._controller

    @staticmethod
    def _resolve_group_zones(
        group_config: config_pb2.ScaleGroupConfig,
        platform_config: config_pb2.PlatformConfig,
        zone_override: str | None,
    ) -> list[str | None]:
        if zone_override:
            return [zone_override]

        if platform_config.WhichOneof("platform") != "gcp":
            return [None]

        gcp = platform_config.gcp
        if group_config.zones:
            return list(group_config.zones)
        if gcp.default_zones:
            return list(gcp.default_zones)
        if gcp.zone:
            return [gcp.zone]
        return [None]
