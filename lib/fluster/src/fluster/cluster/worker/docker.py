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

"""Docker runtime with cgroups v2 resource limits and BuildKit image caching."""

# TODO - set things up some memray/pyspy/etc work as expected
# these need to be installed at least, and then need maybe some permissions

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import docker
import docker.errors

from fluster.rpc import cluster_pb2
from fluster.cluster.worker.worker_types import JobLogs, LogLine


@dataclass
class ContainerConfig:
    image: str
    command: list[str]
    env: dict[str, str]
    workdir: str = "/app"
    resources: cluster_pb2.ResourceSpec | None = None
    timeout_seconds: int | None = None
    mounts: list[tuple[str, str, str]] = field(default_factory=list)  # (host, container, mode)
    ports: dict[str, int] = field(default_factory=dict)  # name -> host_port

    def _parse_memory_mb(self, memory_str: str) -> int:
        """Parse memory string like '8g', '128m' to MB."""
        # TODO humansomething parser
        match = re.match(r"^(\d+)([gmk]?)$", memory_str.lower())
        if not match:
            raise ValueError(f"Invalid memory format: {memory_str}")

        value, unit = match.groups()
        value = int(value)

        if unit == "g":
            return value * 1024
        elif unit == "k":
            return value // 1024
        else:  # 'm' or no unit (assume MB)
            return value

    def get_cpu_millicores(self) -> int | None:
        if not self.resources or not self.resources.cpu:
            return None
        return self.resources.cpu * 1000

    def get_memory_mb(self) -> int | None:
        if not self.resources or not self.resources.memory:
            return None
        return self._parse_memory_mb(self.resources.memory)


@dataclass
class ContainerResult:
    container_id: str
    exit_code: int
    started_at: float
    finished_at: float
    error: str | None = None


@dataclass
class ContainerStats:
    """Parsed container statistics.

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


@dataclass
class ContainerStatus:
    """Container state from docker inspect.

    Attributes:
        running: True if container is currently running
        exit_code: Exit code if container has exited, None if still running
        error: Error message if container failed to start
    """

    running: bool
    exit_code: int | None = None
    error: str | None = None


@dataclass
class ImageInfo:
    tag: str
    created_at: str


# =============================================================================
# Protocols
# =============================================================================


class ContainerRuntime(Protocol):
    """Protocol for container runtimes (Docker, Firecracker, Podman, etc.)."""

    def create_container(self, config: ContainerConfig) -> str: ...

    def start_container(self, container_id: str) -> None: ...

    def inspect(self, container_id: str) -> ContainerStatus: ...

    def kill(self, container_id: str, force: bool = False) -> None: ...

    def remove(self, container_id: str) -> None: ...

    def get_logs(self, container_id: str) -> list[LogLine]: ...

    def get_stats(self, container_id: str) -> ContainerStats: ...


class ImageBuilder(Protocol):
    """Protocol for image building (Docker build, rootfs creation, etc.)."""

    def build(
        self,
        context: Path,
        dockerfile_content: str,
        tag: str,
        job_logs: JobLogs | None = None,
    ) -> None: ...

    def exists(self, tag: str) -> bool: ...

    def remove(self, tag: str) -> None: ...

    def list_images(self, pattern: str) -> list[ImageInfo]: ...


class DockerRuntime:
    """Execute containers via Docker CLI with cgroups v2 resource limits.

    Security hardening: no-new-privileges, cap-drop ALL.
    Uses subprocess for lifecycle, Docker Python SDK for stats/logs.
    """

    def create_container(self, config: ContainerConfig) -> str:
        cmd = [
            "docker",
            "create",
            "--add-host=host.docker.internal:host-gateway",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "-w",
            config.workdir,
        ]

        # Resource limits (cgroups v2)
        cpu_millicores = config.get_cpu_millicores()
        if cpu_millicores:
            cpus = cpu_millicores / 1000
            cmd.extend(["--cpus", str(cpus)])
        memory_mb = config.get_memory_mb()
        if memory_mb:
            cmd.extend(["--memory", f"{memory_mb}m"])

        # Environment variables
        for k, v in config.env.items():
            cmd.extend(["-e", f"{k}={v}"])

        # Mounts
        for host, container, mode in config.mounts:
            cmd.extend(["-v", f"{host}:{container}:{mode}"])

        # Port mappings (name is for reference only, bind host->container)
        for host_port in config.ports.values():
            cmd.extend(["-p", f"{host_port}:{host_port}"])

        cmd.append(config.image)
        cmd.extend(config.command)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")
        return result.stdout.strip()

    def start_container(self, container_id: str) -> None:
        result = subprocess.run(
            ["docker", "start", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

    def inspect(self, container_id: str) -> ContainerStatus:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                container_id,
                "--format",
                "{{json .State}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return ContainerStatus(running=False, error="Container not found")

        try:
            state = json.loads(result.stdout.strip())
            running = state.get("Running", False)
            exit_code = state.get("ExitCode")
            error_msg = state.get("Error", "") or None

            return ContainerStatus(
                running=running,
                exit_code=exit_code if not running else None,
                error=error_msg,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return ContainerStatus(running=False, error=f"Failed to parse inspect output: {e}")

    def kill(self, container_id: str, force: bool = False) -> None:
        signal = "SIGKILL" if force else "SIGTERM"
        result = subprocess.run(
            ["docker", "kill", f"--signal={signal}", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to kill container: {result.stderr}")

    def remove(self, container_id: str) -> None:
        result = subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to remove container: {result.stderr}")

    def get_logs(self, container_id: str) -> list[LogLine]:
        client = docker.from_env()  # type: ignore[attr-defined]
        try:
            container = client.containers.get(container_id)
        except docker.errors.NotFound:
            return []

        logs: list[LogLine] = []

        # Fetch stdout with timestamps
        stdout_logs = container.logs(stdout=True, stderr=False, timestamps=True)
        for line in stdout_logs.decode().splitlines():
            if line:
                timestamp, data = self._parse_docker_log_line(line)
                logs.append(LogLine(timestamp=timestamp, source="stdout", data=data))

        # Fetch stderr with timestamps
        stderr_logs = container.logs(stdout=False, stderr=True, timestamps=True)
        for line in stderr_logs.decode().splitlines():
            if line:
                timestamp, data = self._parse_docker_log_line(line)
                logs.append(LogLine(timestamp=timestamp, source="stderr", data=data))

        return logs

    def _parse_docker_log_line(self, line: str) -> tuple[datetime, str]:
        if len(line) > 30 and line[10] == "T":
            z_idx = line.find("Z")
            if 20 < z_idx < 35:
                ts_str = line[: z_idx + 1]
                # Truncate nanoseconds to microseconds for fromisoformat
                if len(ts_str) > 27:
                    ts_str = ts_str[:26] + "Z"
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    return ts, line[z_idx + 2 :]
                except ValueError:
                    pass
        return datetime.now(timezone.utc), line

    def get_stats(self, container_id: str) -> ContainerStats:
        client = docker.from_env()  # type: ignore[attr-defined]
        try:
            container = client.containers.get(container_id)
            stats = container.stats(stream=False)

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


class DockerImageBuilder:
    """Build Docker images using Docker CLI with BuildKit."""

    def __init__(self, registry: str):
        self._registry = registry

    def build(
        self,
        context: Path,
        dockerfile_content: str,
        tag: str,
        job_logs: JobLogs | None = None,
    ) -> None:
        dockerfile_path = context / "Dockerfile.fluster"
        dockerfile_path.write_text(dockerfile_content)

        try:
            if job_logs:
                job_logs.add("build", f"Starting build for image: {tag}")

            cmd = [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                tag,
                "--build-arg",
                "BUILDKIT_INLINE_CACHE=1",
                str(context),
            ]

            proc = subprocess.Popen(
                cmd,
                env={**os.environ, "DOCKER_BUILDKIT": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output to job_logs
            if proc.stdout:
                for line in proc.stdout:
                    if job_logs:
                        job_logs.add("build", line.rstrip())

            returncode = proc.wait()

            if job_logs:
                if returncode == 0:
                    job_logs.add("build", "Build completed successfully")
                else:
                    job_logs.add("build", f"Build failed with exit code {returncode}")

            if returncode != 0:
                raise RuntimeError(f"Docker build failed with exit code {returncode}")
        finally:
            # Cleanup generated dockerfile
            dockerfile_path.unlink(missing_ok=True)

    def exists(self, tag: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def remove(self, tag: str) -> None:
        subprocess.run(
            ["docker", "rmi", tag],
            capture_output=True,
            check=False,
        )

    def list_images(self, pattern: str) -> list[ImageInfo]:
        result = subprocess.run(
            [
                "docker",
                "images",
                "--format",
                "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
                "--filter",
                f"reference={pattern}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        images = []
        for line in result.stdout.strip().split("\n"):
            if line and "\t" in line:
                tag, created = line.split("\t", 1)
                images.append(ImageInfo(tag=tag, created_at=created))

        return images
