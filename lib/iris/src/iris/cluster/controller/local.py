# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local in-process controller for testing.

This module provides LocalController which runs the controller and autoscaler
in the current process for local testing. Workers are threads, not VMs.

Provides:
- create_local_autoscaler: Factory for creating autoscaler with LocalPlatform
- LocalController: In-process controller implementation for testing
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.controller import (
    Controller as _InnerController,
    ControllerConfig as _InnerControllerConfig,
    RpcWorkerStubFactory,
)
from iris.cluster.controller.lifecycle import ControllerStatus
from iris.cluster.controller.scaling_group import ScalingGroup
from iris.cluster.platform.local import LocalPlatform, find_free_port
from iris.cluster.worker.port_allocator import PortAllocator
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_utils import Duration


def create_local_autoscaler(
    config: config_pb2.IrisClusterConfig,
    controller_address: str,
    threads: ThreadContainer | None = None,
) -> tuple[Autoscaler, tempfile.TemporaryDirectory]:
    """Create Autoscaler with LocalPlatform for all scale groups.

    Creates temp directories and a PortAllocator so that LocalPlatform can
    spawn real Worker threads that register with the controller.

    Args:
        config: Cluster configuration (with defaults already applied)
        controller_address: Address for workers to connect to
        threads: Optional thread container for testing

    Returns:
        Tuple of (autoscaler, temp_dir). The caller owns the temp_dir and
        must call cleanup() when done.
    """
    label_prefix = config.platform.label_prefix or "iris"

    temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_autoscaler_")
    temp_path = Path(temp_dir.name)
    cache_path = temp_path / "cache"
    cache_path.mkdir()
    fake_bundle = temp_path / "fake_bundle"
    fake_bundle.mkdir()
    (fake_bundle / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    port_allocator = PortAllocator()

    platform = LocalPlatform(
        label_prefix=label_prefix,
        threads=threads,
        controller_address=controller_address,
        cache_path=cache_path,
        fake_bundle=fake_bundle,
        port_allocator=port_allocator,
    )

    scale_up_delay = Duration.from_proto(config.defaults.autoscaler.scale_up_delay)
    scale_down_delay = Duration.from_proto(config.defaults.autoscaler.scale_down_delay)

    scale_groups: dict[str, ScalingGroup] = {}
    for name, sg_config in config.scale_groups.items():
        scale_groups[name] = ScalingGroup(
            config=sg_config,
            platform=platform,
            label_prefix=label_prefix,
            scale_up_cooldown=scale_up_delay,
            scale_down_cooldown=scale_down_delay,
            threads=threads,
        )

    autoscaler = Autoscaler.from_config(
        scale_groups=scale_groups,
        config=config.defaults.autoscaler,
        platform=platform,
        threads=threads,
    )
    return autoscaler, temp_dir


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

    Runs Controller + Autoscaler(LocalPlatform) in the current process.
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
        self._autoscaler_temp_dir: tempfile.TemporaryDirectory | None = None

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
        self._autoscaler, self._autoscaler_temp_dir = create_local_autoscaler(
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
        if self._autoscaler is not None:
            self._autoscaler = None
        if self._autoscaler_temp_dir is not None:
            self._autoscaler_temp_dir.cleanup()
            self._autoscaler_temp_dir = None
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
