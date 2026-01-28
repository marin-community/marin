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
