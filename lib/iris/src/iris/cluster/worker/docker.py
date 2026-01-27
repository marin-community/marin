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

import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from iris.rpc import cluster_pb2
from iris.cluster.worker.worker_types import TaskLogs, LogLine


@dataclass
class ContainerConfig:
    """Configuration for running a container.

    The serialized_entrypoint contains cloudpickle-serialized (callable, args, kwargs).
    We pass the bytes through to avoid re-serialization issues with cloudpickle.
    """

    image: str
    serialized_entrypoint: bytes
    env: dict[str, str]
    workdir: str = "/app"
    resources: cluster_pb2.ResourceSpecProto | None = None
    timeout_seconds: int | None = None
    mounts: list[tuple[str, str, str]] = field(default_factory=list)  # (host, container, mode)
    ports: dict[str, int] = field(default_factory=dict)  # name -> host_port
    task_id: str | None = None
    job_id: str | None = None

    def get_cpu_millicores(self) -> int | None:
        if not self.resources or not self.resources.cpu:
            return None
        return self.resources.cpu * 1000

    def get_memory_mb(self) -> int | None:
        if not self.resources or not self.resources.memory_bytes:
            return None
        return self.resources.memory_bytes // (1024 * 1024)


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

    def list_iris_containers(self, all_states: bool = True) -> list[str]: ...

    def remove_all_iris_containers(self) -> int: ...


class ImageBuilder(Protocol):
    """Protocol for image building (Docker build, rootfs creation, etc.)."""

    def build(
        self,
        context: Path,
        dockerfile_content: str,
        tag: str,
        task_logs: TaskLogs | None = None,
    ) -> None: ...

    def exists(self, tag: str) -> bool: ...

    def remove(self, tag: str) -> None: ...

    def list_images(self, pattern: str) -> list[ImageInfo]: ...


class DockerRuntime:
    """Execute containers via Docker CLI with cgroups v2 resource limits.

    Security hardening: no-new-privileges, cap-drop ALL.
    Uses subprocess for lifecycle, Docker Python SDK for stats/logs.
    """

    def _build_command(self, serialized_entrypoint: bytes) -> list[str]:
        """Build the Python command that executes the entrypoint in a container.

        Takes already-serialized bytes to avoid re-serialization issues with cloudpickle.
        The bytes contain a cloudpickle-serialized (callable, args, kwargs) tuple.
        We serialize as a tuple (not an Entrypoint class) because job containers
        don't have iris installed.

        Entrypoint format: The serialized bytes decode to (callable, args, kwargs) tuple.
        This format is produced by Entrypoint.serialize() in iris.cluster.types.
        We deserialize directly here rather than using Entrypoint.deserialize() because
        job containers don't have iris installed.
        """
        encoded = base64.b64encode(serialized_entrypoint).decode()

        thunk = f"""
import cloudpickle
import base64
import sys
import traceback

try:
    fn, args, kwargs = cloudpickle.loads(base64.b64decode('{encoded}'))
    result = fn(*args, **kwargs)

    with open('/tmp/_result.pkl', 'wb') as f:
        f.write(cloudpickle.dumps(result))
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
        return ["python", "-u", "-c", thunk]

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

        # Add iris labels for discoverability
        cmd.extend(["--label", "iris.managed=true"])
        if config.task_id:
            cmd.extend(["--label", f"iris.task_id={config.task_id}"])
        if config.job_id:
            cmd.extend(["--label", f"iris.job_id={config.job_id}"])

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
        cmd.extend(self._build_command(config.serialized_entrypoint))

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
        status = self.inspect(container_id)
        if not status.running:
            return

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
        logs: list[LogLine] = []

        # Fetch stdout with timestamps
        result = subprocess.run(
            ["docker", "logs", "--timestamps", container_id],
            capture_output=True,
            text=True,
            check=False,
        )

        # Container not found
        if result.returncode != 0:
            return []

        # Docker logs combines stdout and stderr by default, but we can't easily separate them
        # after the fact. We'll use stderr redirection to get them separately.
        stdout_result = subprocess.run(
            ["docker", "logs", "--timestamps", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )

        for line in stdout_result.stdout.splitlines():
            if line:
                timestamp, data = self._parse_docker_log_line(line)
                logs.append(LogLine(timestamp=timestamp, source="stdout", data=data))

        # Fetch stderr with timestamps
        stderr_result = subprocess.run(
            ["docker", "logs", "--timestamps", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        for line in stderr_result.stderr.splitlines():
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
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_id],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return ContainerStats(
                memory_mb=0,
                cpu_percent=0,
                process_count=0,
                available=False,
            )

        try:
            stats = json.loads(result.stdout.strip())

            # Parse memory usage (format: "123.4MiB / 2GiB")
            memory_str = stats.get("MemUsage", "0B / 0B").split("/")[0].strip()
            memory_mb = self._parse_memory_size(memory_str)

            # Parse CPU percentage (format: "12.34%")
            cpu_str = stats.get("CPUPerc", "0%").rstrip("%")
            cpu_percent = int(float(cpu_str)) if cpu_str else 0

            # Parse process count (format: "5")
            pids_str = stats.get("PIDs", "0")
            process_count = int(pids_str) if pids_str.isdigit() else 0

            return ContainerStats(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                process_count=process_count,
                available=True,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # Log parsing error but return unavailable stats
            return ContainerStats(
                memory_mb=0,
                cpu_percent=0,
                process_count=0,
                available=False,
            )

    def _parse_memory_size(self, size_str: str) -> int:
        """Parse memory size string like '123.4MiB' to MB."""
        size_str = size_str.strip()
        match = re.match(r"^([\d.]+)\s*([KMGT]i?B?)$", size_str, re.IGNORECASE)
        if not match:
            return 0

        value = float(match.group(1))
        unit = match.group(2).upper()

        # Convert to MB
        if unit.startswith("K"):
            return int(value / 1024)
        elif unit.startswith("M"):
            return int(value)
        elif unit.startswith("G"):
            return int(value * 1024)
        elif unit.startswith("T"):
            return int(value * 1024 * 1024)
        elif unit == "B":
            return int(value / (1024 * 1024))
        else:
            return 0

    def list_iris_containers(self, all_states: bool = True) -> list[str]:
        """List all containers with iris.managed=true label."""
        cmd = ["docker", "ps", "-q", "--filter", "label=iris.managed=true"]
        if all_states:
            cmd.append("-a")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return []
        return [cid for cid in result.stdout.strip().split("\n") if cid]

    def remove_all_iris_containers(self) -> int:
        """Force remove all iris-managed containers. Returns count attempted."""
        container_ids = self.list_iris_containers(all_states=True)
        if not container_ids:
            return 0

        subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            capture_output=True,
            check=False,
        )
        return len(container_ids)


class DockerImageBuilder:
    """Build Docker images using Docker CLI with BuildKit."""

    def __init__(self, registry: str):
        self._registry = registry

    def build(
        self,
        context: Path,
        dockerfile_content: str,
        tag: str,
        task_logs: TaskLogs | None = None,
    ) -> None:
        dockerfile_path = context / "Dockerfile.iris"
        dockerfile_path.write_text(dockerfile_content)

        try:
            if task_logs:
                task_logs.add("build", f"Starting build for image: {tag}")

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

            # Stream output to task_logs
            if proc.stdout:
                for line in proc.stdout:
                    if task_logs:
                        task_logs.add("build", line.rstrip())

            returncode = proc.wait()

            if task_logs:
                if returncode == 0:
                    task_logs.add("build", "Build completed successfully")
                else:
                    task_logs.add("build", f"Build failed with exit code {returncode}")

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
