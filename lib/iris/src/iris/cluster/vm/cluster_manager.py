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
from collections.abc import Iterator
from contextlib import contextmanager

from iris.cluster.vm.controller import ControllerProtocol, create_controller
from iris.cluster.vm.debug import controller_tunnel
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)


def make_local_config(
    base_config: config_pb2.IrisClusterConfig,
) -> config_pb2.IrisClusterConfig:
    """Override a GCP/manual config to run locally.

    Replaces the controller oneof with LocalControllerConfig and every
    scale group's provider oneof with LocalProvider. Everything else
    (accelerator_type, accelerator_variant, min/max_slices) is preserved.
    """
    config = config_pb2.IrisClusterConfig()
    config.CopyFrom(base_config)
    config.controller_vm.ClearField("controller")
    config.controller_vm.local.port = 0  # auto-assign
    config.controller_vm.bundle_prefix = ""  # LocalController will set temp path
    for sg in config.scale_groups.values():
        sg.provider.ClearField("provider")
        sg.provider.local.SetInParent()
    # Local mode needs fast autoscaler evaluation for tests
    if not config.autoscaler.evaluation_interval_seconds:
        config.autoscaler.evaluation_interval_seconds = 0.5
    return config


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

    def __init__(self, config: config_pb2.IrisClusterConfig):
        self._config = config
        self._controller: ControllerProtocol | None = None

    @property
    def is_local(self) -> bool:
        return self._config.controller_vm.WhichOneof("controller") == "local"

    def start(self) -> str:
        """Start the controller. Returns the controller address.

        For GCP: creates a GCE VM, bootstraps, returns internal IP.
        For local: starts in-process Controller, returns localhost URL.
        """
        self._controller = create_controller(self._config)
        address = self._controller.start()
        logger.info("Controller started at %s (local=%s)", address, self.is_local)
        return address

    def stop(self) -> None:
        """Stop the controller and clean up resources."""
        if self._controller:
            self._controller.stop()
            self._controller = None
            logger.info("Controller stopped")

    def reload(self) -> str:
        """Reload cluster: redeploy containers on existing VMs.

        Reloads controller and all workers by re-running bootstrap scripts
        without recreating VMs. Much faster than stop/start cycle.

        For GCP: SSHs into existing VMs and pulls new images, restarts containers.
        For local: Equivalent to restart (no VMs to preserve).

        Returns:
            Controller address

        Raises:
            RuntimeError: If controller or worker VMs don't exist
        """
        # Reload controller
        self._controller = create_controller(self._config)
        address = self._controller.reload()
        logger.info("Controller reloaded at %s", address)

        # Reload workers (GCP only - local workers restart with controller)
        if not self.is_local:
            self._reload_workers()

        return address

    def _reload_workers(self) -> None:
        """Reload all worker VMs by discovering and re-running bootstrap."""
        from iris.cluster.vm.config import _create_manager_from_config
        from iris.cluster.vm.managed_vm import TrackedVmFactory, VmRegistry

        # Create temporary registry and factory for discovery
        vm_registry = VmRegistry()
        vm_factory = TrackedVmFactory(vm_registry)

        logger.info("Reloading workers across %d scale group(s)", len(self._config.scale_groups))

        for group_name in self._config.scale_groups:
            logger.info("Processing scale group: %s", group_name)

            # Create VmManager for this scale group
            manager = _create_manager_from_config(
                group_name=group_name,
                cluster_config=self._config,
                vm_factory=vm_factory,
                dry_run=False,
            )

            # Discover existing VMs
            vm_groups = manager.discover_vm_groups()

            if not vm_groups:
                logger.warning("No existing VMs found for scale group %s", group_name)
                continue

            logger.info("Found %d VM group(s) in %s", len(vm_groups), group_name)

            # Reload each VM in each group
            for group in vm_groups:
                logger.info("Reloading VM group %s", group.group_id)
                for vm in group.vms():
                    vm.reload()
                    logger.info("  ✓ Reloaded VM %s", vm.info.vm_id)

        logger.info("Worker reload complete")

    @contextmanager
    def connect(self) -> Iterator[str]:
        """Start controller, yield a usable URL, stop on exit.

        For GCP: establishes SSH tunnel, yields tunnel URL.
        For local: yields direct localhost URL (no tunnel).
        """
        address = self.start()
        try:
            if self.is_local:
                yield address
            else:
                zone = self._config.zone
                project = self._config.project_id
                label_prefix = self._config.label_prefix or "iris"
                with controller_tunnel(zone, project, label_prefix=label_prefix) as tunnel_url:
                    yield tunnel_url
        finally:
            self.stop()

    @property
    def controller(self) -> ControllerProtocol:
        """Access the underlying controller (must call start() first)."""
        if self._controller is None:
            raise RuntimeError("ClusterManager.start() not called")
        return self._controller
