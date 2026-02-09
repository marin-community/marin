# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster lifecycle manager.

Provides a uniform interface for starting/stopping/connecting to an Iris
cluster regardless of backend (GCP, manual, local). Callers get a URL;
ClusterManager handles tunnel setup, mode detection, and cleanup.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from iris.cluster.vm.config import IrisConfig
from iris.cluster.vm.controller_vm import ControllerProtocol, create_controller_vm
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)


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
    ):
        self._config = config
        self._threads = threads if threads is not None else get_thread_container()
        self._controller: ControllerProtocol | None = None

    @property
    def is_local(self) -> bool:
        return self._config.controller.WhichOneof("controller") == "local"

    def start(self) -> str:
        """Start the controller. Returns the controller address.

        For GCP: creates a GCE VM, bootstraps, returns internal IP.
        For local: starts in-process Controller, returns localhost URL.
        """
        self._controller = create_controller_vm(self._config, threads=self._threads)
        address = self._controller.start()
        logger.info("Controller started at %s (local=%s)", address, self.is_local)
        return address

    def stop(self) -> None:
        """Stop the controller and clean up resources.

        Shutdown ordering:
        1. Stop the controller (which stops its threads and autoscaler)
        2. Wait on the root ThreadContainer to verify all threads have exited
        """
        if self._controller:
            self._controller.stop()
            self._controller = None
            logger.info("Controller stopped")

        self._threads.wait()

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
        # Reload controller - it will re-bootstrap workers on startup
        self._controller = create_controller_vm(self._config, threads=self._threads)
        address = self._controller.reload()
        logger.info("Controller reloaded at %s", address)

        return address

    @contextmanager
    def connect(self) -> Iterator[str]:
        """Start controller, yield a usable URL, stop on exit.

        For GCP: establishes SSH tunnel, yields tunnel URL.
        For local: yields direct localhost URL (no tunnel).
        """
        address = self.start()
        try:
            # Use Platform.tunnel() for consistent connection handling
            iris_config = IrisConfig(self._config)
            platform = iris_config.platform()
            with platform.tunnel(address) as tunnel_url:
                yield tunnel_url
        finally:
            self.stop()

    @property
    def controller(self) -> ControllerProtocol:
        """Access the underlying controller (must call start() first)."""
        if self._controller is None:
            raise RuntimeError("ClusterManager.start() not called")
        return self._controller
