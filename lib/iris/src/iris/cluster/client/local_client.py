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

"""Local cluster client using real Controller/Worker with in-process execution.

Spins up a real Controller and Worker but executes jobs in-process using threads
instead of Docker containers, ensuring local execution follows the same code path
as production cluster execution.
"""

import time
from typing import Self

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.types import Entrypoint
from iris.cluster.vm.cluster_manager import ClusterManager
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync


def _make_local_cluster_config(max_workers: int) -> config_pb2.IrisClusterConfig:
    """Build a fully-configured IrisClusterConfig for local execution.

    Sets up controller_vm.local, bundle_prefix, scale groups with local provider,
    and fast autoscaler evaluation for tests.
    """
    config = config_pb2.IrisClusterConfig()

    # Configure local controller
    config.controller_vm.local.port = 0  # auto-assign
    config.controller_vm.bundle_prefix = ""  # LocalController will set temp path

    # Configure scale group with local provider
    sg = config_pb2.ScaleGroupConfig(
        name="local-cpu",
        min_slices=1,
        max_slices=max_workers,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    sg.provider.local.SetInParent()
    config.scale_groups["local-cpu"].CopyFrom(sg)

    # Fast autoscaler evaluation for tests
    config.autoscaler.evaluation_interval_seconds = 0.5

    return config


class LocalClusterClient:
    """Local cluster client using real Controller/Worker with in-process execution.

    Provides the same execution path as production clusters while running
    entirely in-process without Docker or network dependencies.

    Use the create() classmethod to instantiate:
        client = LocalClusterClient.create()
        # ... use client ...
        client.shutdown()
    """

    def __init__(self, manager: ClusterManager, remote_client: RemoteClusterClient):
        self._manager = manager
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
        manager = ClusterManager(config)
        address = manager.start()
        cls._wait_for_worker_registration(address)
        remote_client = RemoteClusterClient(controller_address=address, timeout_ms=30000)
        return cls(manager, remote_client)

    @staticmethod
    def _wait_for_worker_registration(controller_address: str, timeout: float = 10.0) -> None:
        temp_client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=30000,
        )
        try:
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                response = temp_client.list_workers(cluster_pb2.Controller.ListWorkersRequest())
                if response.workers:
                    return
                time.sleep(0.1)
            raise TimeoutError("Worker failed to register with controller")
        finally:
            temp_client.close()

    def shutdown(self, wait: bool = True) -> None:
        del wait
        self._remote_client.shutdown()
        self._manager.stop()

    def submit_job(
        self,
        job_id: str,
        entrypoint: Entrypoint,
        resources: cluster_pb2.ResourceSpecProto,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
        constraints: list[cluster_pb2.Constraint] | None = None,
        coscheduling: cluster_pb2.CoschedulingConfig | None = None,
    ) -> None:
        self._remote_client.submit_job(
            job_id=job_id,
            entrypoint=entrypoint,
            resources=resources,
            environment=environment,
            ports=ports,
            scheduling_timeout_seconds=scheduling_timeout_seconds,
            constraints=constraints,
            coscheduling=coscheduling,
        )

    def get_job_status(self, job_id: str) -> cluster_pb2.JobStatus:
        return self._remote_client.get_job_status(job_id)

    def wait_for_job(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus:
        return self._remote_client.wait_for_job(job_id, timeout=timeout, poll_interval=poll_interval)

    def terminate_job(self, job_id: str) -> None:
        self._remote_client.terminate_job(job_id)

    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        return self._remote_client.register_endpoint(name=name, address=address, job_id=job_id, metadata=metadata)

    def unregister_endpoint(self, endpoint_id: str) -> None:
        self._remote_client.unregister_endpoint(endpoint_id)

    def list_endpoints(self, prefix: str) -> list[cluster_pb2.Controller.Endpoint]:
        return self._remote_client.list_endpoints(prefix)

    def list_jobs(self) -> list[cluster_pb2.JobStatus]:
        return self._remote_client.list_jobs()

    def get_task_status(self, job_id: str, task_index: int) -> cluster_pb2.TaskStatus:
        return self._remote_client.get_task_status(job_id, task_index)

    def list_tasks(self, job_id: str) -> list[cluster_pb2.TaskStatus]:
        return self._remote_client.list_tasks(job_id)

    def fetch_task_logs(
        self,
        task_id: str,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[cluster_pb2.Worker.LogEntry]:
        return self._remote_client.fetch_task_logs(task_id, start_ms, max_lines)
