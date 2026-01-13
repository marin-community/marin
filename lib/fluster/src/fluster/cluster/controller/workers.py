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

"""Worker registry and scheduling for the controller.

This module provides:
- WorkerConfig: Static worker configuration for v0
- load_workers_from_config(): Register workers from static config at startup
- find_worker_for_job(): Simple first-fit worker selection for v0

Future versions will add resource matching (TPU type, memory, etc.) to
find_worker_for_job().
"""

import time
from dataclasses import dataclass

from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import WorkerId


@dataclass
class WorkerConfig:
    """Static worker configuration for v0.

    Args:
        worker_id: Unique worker identifier
        address: Worker RPC address (host:port)
        resources: Worker's available resources
    """

    worker_id: str
    address: str
    resources: cluster_pb2.ResourceSpec


def load_workers_from_config(
    state: ControllerState,
    workers: list[WorkerConfig],
) -> None:
    """Register workers from static config.

    Creates ControllerWorker instances from the provided config and adds them
    to the controller state. Sets last_heartbeat_ms to the current time so
    workers are considered healthy at startup.

    Args:
        state: Controller state to update
        workers: List of worker configurations to register
    """
    now_ms = int(time.time() * 1000)

    for cfg in workers:
        worker = ControllerWorker(
            worker_id=WorkerId(cfg.worker_id),
            address=cfg.address,
            resources=cfg.resources,
            last_heartbeat_ms=now_ms,
        )
        state.add_worker(worker)


def find_worker_for_job(
    state: ControllerState,
    job: ControllerJob,
) -> ControllerWorker | None:
    """Find a worker that can run the given job.

    For v0: simple first-fit on available workers. Returns the first healthy
    worker found, without checking resource compatibility.

    Future: will match resources (TPU type, memory, etc.) to job requirements.

    Args:
        state: Controller state containing worker registry
        job: Job to find a worker for (unused in v0, will be used for resource matching)

    Returns:
        First healthy worker if available, None otherwise
    """
    workers = state.get_available_workers()
    for worker in workers:
        # For v0, any healthy worker works
        # Future: Check resource compatibility with job.request.resources
        return worker
    return None
