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

"""Worker registry and resource-aware scheduling utilities."""

import time
from dataclasses import dataclass

from fluster.rpc import cluster_pb2
from fluster.cluster.controller.resources import (
    get_device_type,
    get_device_variant,
    get_gpu_count,
    parse_memory_string,
)
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
    now_ms = int(time.time() * 1000)

    for cfg in workers:
        worker = ControllerWorker(
            worker_id=WorkerId(cfg.worker_id),
            address=cfg.address,
            resources=cfg.resources,
            last_heartbeat_ms=now_ms,
        )
        state.add_worker(worker)


def get_committed_resources(
    state: ControllerState,
    worker: ControllerWorker,
) -> tuple[int, int, int]:
    """Compute resources committed to running jobs on this worker."""
    cpu = 0
    memory = 0
    gpu = 0

    for job_id in worker.running_jobs:
        job = state.get_job(job_id)
        if job:
            resources = job.request.resources
            cpu += resources.cpu
            memory += parse_memory_string(resources.memory)
            gpu += get_gpu_count(resources.device)

    return cpu, memory, gpu


def worker_can_fit_job(
    state: ControllerState,
    worker: ControllerWorker,
    job: ControllerJob,
    additional_jobs: list[ControllerJob] | None = None,
) -> bool:
    """Check if worker has sufficient available capacity for job.

    Checks CPU, memory, device type, device variant, and GPU count.
    """
    job_resources = job.request.resources
    worker_resources = worker.resources

    # Get committed resources dynamically from running jobs
    committed_cpu, committed_memory, committed_gpu = get_committed_resources(state, worker)

    # Add resources from jobs assigned this round but not yet dispatched
    if additional_jobs:
        for additional_job in additional_jobs:
            add_resources = additional_job.request.resources
            committed_cpu += add_resources.cpu
            committed_memory += parse_memory_string(add_resources.memory)
            committed_gpu += get_gpu_count(add_resources.device)

    # CPU check
    available_cpu = worker_resources.cpu - committed_cpu
    if job_resources.cpu > available_cpu:
        return False

    # Memory check
    worker_memory = parse_memory_string(worker_resources.memory)
    job_memory = parse_memory_string(job_resources.memory)
    available_memory = worker_memory - committed_memory
    if job_memory > available_memory:
        return False

    # Device type check
    job_device_type = get_device_type(job_resources.device)
    worker_device_type = get_device_type(worker_resources.device)

    if job_device_type != worker_device_type:
        return False

    # Device variant check (only if job specifies one that's not "auto")
    job_variant = get_device_variant(job_resources.device)
    if job_variant and job_variant != "auto":
        worker_variant = get_device_variant(worker_resources.device)
        if worker_variant != job_variant:
            return False

    # GPU count check
    if job_device_type == "gpu":
        job_gpu_count = get_gpu_count(job_resources.device)
        worker_gpu_count = get_gpu_count(worker_resources.device)
        available_gpus = worker_gpu_count - committed_gpu
        if job_gpu_count > available_gpus:
            return False

    return True


def find_worker_for_job(
    state: ControllerState,
    job: ControllerJob,
) -> ControllerWorker | None:
    """Find a worker that can run the given job using first-fit strategy."""
    workers = state.get_available_workers()
    for worker in workers:
        if worker_can_fit_job(state, worker, job):
            return worker
    return None
