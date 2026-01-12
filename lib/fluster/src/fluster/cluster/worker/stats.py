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

"""Docker container resource statistics collection."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import docker
import docker.errors


@dataclass
class ContainerStats:
    """Parsed Docker container statistics.

    Attributes:
        memory_mb: Memory usage in megabytes
        cpu_percent: CPU usage as percentage (0-100)
        process_count: Number of processes running in container
        available: False if container stopped or unavailable
    """

    memory_mb: int
    cpu_percent: int
    process_count: int
    available: bool


async def collect_container_stats(docker_client: docker.DockerClient, container_id: str) -> ContainerStats:
    """Collect resource usage from Docker container.

    Uses container.stats(decode=True, stream=False) for single snapshot.

    Parses:
    - memory_stats.usage → convert bytes to MB
    - CPU percentage from cpu_stats/precpu_stats deltas
    - pids_stats.current for process count

    Args:
        docker_client: Docker client instance
        container_id: Container ID to collect stats from

    Returns:
        ContainerStats with available=False on docker.errors.NotFound/APIError
    """
    try:
        # Run blocking Docker operation in thread pool
        container = await asyncio.to_thread(docker_client.containers.get, container_id)
        stats = await asyncio.to_thread(container.stats, decode=True, stream=False)

        # Parse memory usage (bytes to MB)
        memory_bytes = stats.get("memory_stats", {}).get("usage", 0)
        memory_mb = int(memory_bytes / (1024 * 1024))

        # Calculate CPU percentage from deltas
        cpu_percent = _calculate_cpu_percent(stats)

        # Parse process count
        process_count = stats.get("pids_stats", {}).get("current", 0)

        return ContainerStats(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            process_count=process_count,
            available=True,
        )
    except (docker.errors.NotFound, docker.errors.APIError):
        return ContainerStats(
            memory_mb=0,
            cpu_percent=0,
            process_count=0,
            available=False,
        )


def _calculate_cpu_percent(stats: dict) -> int:
    """Calculate CPU percentage from stats deltas.

    Docker stats format provides cpu_stats and precpu_stats for delta calculation.
    CPU percentage = (cpu_delta / system_delta) * num_cpus * 100
    """
    cpu_stats = stats.get("cpu_stats", {})
    precpu_stats = stats.get("precpu_stats", {})

    cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu_stats.get("cpu_usage", {}).get(
        "total_usage", 0
    )
    system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get("system_cpu_usage", 0)

    if system_delta == 0 or cpu_delta == 0:
        return 0

    num_cpus = cpu_stats.get("online_cpus", 1)
    cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0

    return int(cpu_percent)


async def collect_workdir_size_mb(workdir: Path) -> int:
    """Calculate workdir size in MB using du -sm command.

    Args:
        workdir: Path to directory to measure

    Returns:
        Directory size in megabytes, or 0 if directory doesn't exist
    """
    if not workdir.exists():
        return 0

    proc = await asyncio.create_subprocess_exec(
        "du",
        "-sm",
        str(workdir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, _ = await proc.communicate()

    if proc.returncode != 0:
        return 0

    # du -sm output format: "SIZE\tPATH"
    output = stdout.decode().strip()
    size_str = output.split("\t")[0]

    return int(size_str)
