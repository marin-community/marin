"""
Script that launches a configurable jobs on Ray cluster to verify that they
run, see how long they run, see what environment they're running on, etc.
It's a good way to understand empirically what happens when you run Ray tasks,
to benchmark performance, and to stress test.
"""

import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any

import draccus
import fsspec
import psutil
import ray

from marin.utils import fsspec_glob

logger = logging.getLogger("ray")


@dataclass(frozen=True)
class TaskConfig:
    task_id: int

    task_duration: int
    """Time to run this task for (in seconds)."""

    input_path: str | None
    """Path to a file to read."""


@dataclass(frozen=True)
class TaskResult:
    hostname: str
    working_path: str
    num_bytes: int
    read_time: float
    read_speed: float
    memory_total: int
    memory_available: int
    disk_total: int
    disk_free: int


@dataclass(frozen=True)
class BenchmarkConfig:
    num_tasks: int = 10
    """Number of tasks to run in parallel."""

    min_task_duration: int = 1
    """Minimum time to run each task for (in seconds)."""

    max_task_duration: int = 1
    """Maximum time to run each task for (in seconds)."""

    input_path: str | None = None
    """Path to a directory with files to read."""

    results_path: str | None = None
    """Path to write the benchmarking results."""


@dataclass(frozen=True)
class BenchmarkResult:
    num_nodes: int
    resources: dict[str, Any]
    files: list[str] | None
    task_results: list[TaskResult]


@ray.remote
def process(config: TaskConfig) -> TaskResult:
    hostname = socket.gethostname()

    logger.info(f"Task {config.task_id} starting on {hostname}")

    # Read input file
    num_bytes = 0
    read_time = None
    read_speed = None
    if config.input_path:
        start_time = time.time()
        try:
            with fsspec.open(config.input_path, "rb") as f:
                while True:
                    data = f.read(1024 * 1024)
                    if not data:
                        break
                    num_bytes += len(data)
        except Exception as e:
            logger.error(f"Error reading {config.input_path}: {e}")
        end_time = time.time()
        read_time = end_time - start_time
        read_speed = num_bytes / read_time

    time.sleep(config.task_duration)

    # Get information
    working_path = os.getcwd()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(".")

    logger.info(f"Task {config.task_id} finished on {hostname}")

    return TaskResult(
        hostname=hostname,
        working_path=working_path,
        num_bytes=num_bytes,
        read_time=read_time,
        read_speed=read_speed,
        memory_total=memory.total,
        memory_available=memory.available,
        disk_total=disk.total,
        disk_free=disk.free,
    )


@draccus.wrap()
def main(config: BenchmarkConfig):
    logger.info("Benchmark starting")
    logger.info(f"  {len(ray.nodes())} nodes")
    logger.info(f"  {ray.cluster_resources()} resources")

    num_tasks = config.num_tasks

    files = None
    if config.input_path:
        files = fsspec_glob(os.path.join(config.input_path, "**/*.*"))
        if num_tasks == -1:
            num_tasks = len(files)

    task_configs = []
    for i in range(num_tasks):
        task_duration = config.min_task_duration + i % (config.max_task_duration + 1 - config.min_task_duration)
        input_path = files[i % len(files)] if files else None
        task_configs.append(
            TaskConfig(
                task_id=i,
                task_duration=task_duration,
                input_path=input_path,
            )
        )

    # Execute and get results
    output_refs = list(map(process.remote, task_configs))
    task_results = ray.get(output_refs)

    benchmark_result = BenchmarkResult(
        num_nodes=len(ray.nodes()),
        resources=ray.cluster_resources(),
        files=files,
        task_results=task_results,
    )

    if config.results_path:
        with fsspec.open(config.results_path, "w") as f:
            print(json.dumps(asdict(benchmark_result), indent=2), file=f)
    logger.info("Benchmark complete")


if __name__ == "__main__":
    ray.init()
    main()
