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
- get_committed_resources(): Compute resources committed to running jobs
- worker_can_fit_job(): Check if worker has capacity for a job
- find_worker_for_job(): First-fit worker selection with resource matching
"""

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


def get_committed_resources(
    state: ControllerState,
    worker: ControllerWorker,
) -> tuple[int, int, int]:
    """Compute resources committed to running jobs on this worker.

    Dynamically sums resources from all jobs in worker.running_jobs.
    This approach avoids tracking committed resources incrementally and
    prevents sync issues.

    Args:
        state: Controller state to look up jobs
        worker: Worker to compute committed resources for

    Returns:
        Tuple of (cpu_cores, memory_bytes, gpu_count)
    """
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

    Computes available headroom dynamically from running_jobs plus any
    additional jobs that have been assigned in the current scheduling round
    but not yet dispatched.

    Checks:
    1. CPU: job.cpu <= worker.total_cpu - committed_cpu
    2. Memory: job.memory <= worker.total_memory - committed_memory
    3. Device type: exact match (GPU job only on GPU worker)
    4. Device variant: if job specifies variant (not "auto"), worker must match
    5. GPU count: job.gpu_count <= available_gpus

    Args:
        state: Controller state for job lookups
        worker: Worker to check capacity
        job: Job with resource requirements
        additional_jobs: Jobs assigned this scheduling round but not yet
                        reflected in worker.running_jobs

    Returns:
        True if worker can fit the job
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
    """Find a worker that can run the given job.

    Returns the first healthy worker with sufficient capacity and matching
    device type/variant. Uses first-fit strategy.

    Args:
        state: Controller state containing worker registry
        job: Job to find a worker for

    Returns:
        First matching worker, or None if no worker can fit the job
    """
    workers = state.get_available_workers()
    for worker in workers:
        if worker_can_fit_job(state, worker, job):
            return worker
    return None
