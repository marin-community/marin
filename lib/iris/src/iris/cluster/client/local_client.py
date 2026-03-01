# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local cluster client using real Controller/Worker with in-process execution.

Spins up a real Controller and Worker but executes jobs in-process using threads
instead of Docker containers, ensuring local execution follows the same code path
as production cluster execution.
"""

from typing import Any, Self

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.config import make_local_config
from iris.cluster.controller.local import LocalController
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff


def _make_local_cluster_config(max_workers: int) -> config_pb2.IrisClusterConfig:
    """Build a fully-configured IrisClusterConfig for local execution.

    Creates a minimal base config and transforms it via make_local_config()
    to ensure local defaults (fast autoscaler timings, etc.) are applied
    consistently from config.py.
    """
    # Build minimal base config
    base_config = config_pb2.IrisClusterConfig()

    # Configure scale group (will be transformed to VM_TYPE_LOCAL_VM by make_local_config)
    sg = config_pb2.ScaleGroupConfig(
        name="local-cpu",
        min_slices=1,
        max_slices=max_workers,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            disk_bytes=50 * 1024**3,
            gpu_count=0,
            tpu_count=0,
        ),
    )
    base_config.scale_groups["local-cpu"].CopyFrom(sg)

    # Transform to local config - applies local platform, controller, and fast autoscaler timings
    return make_local_config(base_config)


class LocalClusterClient:
    """Local cluster client using real Controller/Worker with subprocess-based task execution.

    Provides the same execution path as production clusters. Workers run in-process,
    but tasks execute in subprocesses (not Docker containers) for isolation.

    All ClusterClient methods are delegated to the underlying RemoteClusterClient
    via __getattr__. Only lifecycle methods (create, shutdown) are defined here.

    Use the create() classmethod to instantiate:
        client = LocalClusterClient.create()
        # ... use client ...
        client.shutdown()
    """

    def __init__(self, controller: LocalController, remote_client: RemoteClusterClient):
        self._controller = controller
        self._remote_client = remote_client

    @classmethod
    def create(cls, max_workers: int = 4) -> Self:
        """Create and start a local cluster client.

        Args:
            max_workers: Maximum concurrent job threads

        Returns:
            A fully initialized LocalClusterClient ready for use
        """
        config = _make_local_cluster_config(max_workers)
        controller = LocalController(config)
        address = controller.start()
        cls._wait_for_worker_registration(address)
        remote_client = RemoteClusterClient(controller_address=address, timeout_ms=30000)
        return cls(controller, remote_client)

    @staticmethod
    def _wait_for_worker_registration(controller_address: str, timeout: float = 10.0) -> None:
        temp_client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=30000,
        )
        try:
            ExponentialBackoff(initial=0.1, maximum=2.0).wait_until_or_raise(
                lambda: bool(temp_client.list_workers(cluster_pb2.Controller.ListWorkersRequest()).workers),
                timeout=Duration.from_seconds(timeout),
                error_message="Worker failed to register with controller",
            )
        finally:
            temp_client.close()

    def shutdown(self, wait: bool = True) -> None:
        del wait
        self._remote_client.shutdown()
        self._controller.stop()

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the underlying RemoteClusterClient."""
        return getattr(self._remote_client, name)
