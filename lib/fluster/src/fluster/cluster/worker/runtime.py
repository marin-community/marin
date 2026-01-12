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

"""Execute jobs in Docker containers with cgroups v2 resource limits."""

import asyncio
import re
import time
from dataclasses import dataclass, field

from fluster import cluster_pb2


@dataclass
class ContainerConfig:
    """Configuration for container execution."""

    image: str
    command: list[str]
    env: dict[str, str]
    workdir: str = "/app"
    resources: cluster_pb2.ResourceSpec | None = None
    timeout_seconds: int | None = None
    mounts: list[tuple[str, str, str]] = field(default_factory=list)  # (host, container, mode)
    ports: dict[str, int] = field(default_factory=dict)  # name -> host_port
    stdout_file: str | None = None  # Host path for stdout redirection
    stderr_file: str | None = None  # Host path for stderr redirection

    def _parse_memory_mb(self, memory_str: str) -> int:
        """Parse memory string like '8g', '128m' to MB."""
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
        """Get CPU millicores from ResourceSpec."""
        if not self.resources or not self.resources.cpu:
            return None
        return self.resources.cpu * 1000  # Convert cores to millicores

    def get_memory_mb(self) -> int | None:
        """Get memory in MB from ResourceSpec."""
        if not self.resources or not self.resources.memory:
            return None
        return self._parse_memory_mb(self.resources.memory)


@dataclass
class ContainerResult:
    """Result of container execution."""

    container_id: str
    exit_code: int
    started_at: float
    finished_at: float
    error: str | None = None


class DockerRuntime:
    """Execute jobs in Docker containers with cgroups v2 resource limits.

    Requires cgroups v2 (no v1 fallback). Security hardening:
    - no-new-privileges
    - cap-drop ALL
    """

    async def run(
        self,
        config: ContainerConfig,
    ) -> ContainerResult:
        """Run container to completion.

        Args:
            config: Container configuration

        Returns:
            ContainerResult with exit code and timing
        """
        container_id = await self._create_container(config)
        started_at = time.time()

        try:
            await self._start_container(container_id)

            # Wait for completion with timeout
            exit_code = await self._wait_container(
                container_id,
                timeout=config.timeout_seconds,
            )

            return ContainerResult(
                container_id=container_id,
                exit_code=exit_code,
                started_at=started_at,
                finished_at=time.time(),
            )
        except asyncio.TimeoutError:
            await self.kill(container_id, force=True)
            return ContainerResult(
                container_id=container_id,
                exit_code=-1,
                started_at=started_at,
                finished_at=time.time(),
                error="Timeout exceeded",
            )

    async def _create_container(self, config: ContainerConfig) -> str:
        """Create container with cgroups v2 resource limits."""
        cmd = [
            "docker",
            "create",
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
        for _name, host_port in config.ports.items():
            cmd.extend(["-p", f"{host_port}:{host_port}"])

        # If log files specified, mount them and wrap command with redirection
        final_command = config.command
        if config.stdout_file or config.stderr_file:
            # Mount log file directory
            from pathlib import Path

            if config.stdout_file:
                stdout_path = Path(config.stdout_file)
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.touch(exist_ok=True)
                cmd.extend(["-v", f"{stdout_path}:/logs/STDOUT:rw"])

            if config.stderr_file:
                stderr_path = Path(config.stderr_file)
                stderr_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_path.touch(exist_ok=True)
                cmd.extend(["-v", f"{stderr_path}:/logs/STDERR:rw"])

            # Wrap command in shell with redirection
            original_cmd = " ".join(f"'{arg}'" if " " in arg else arg for arg in config.command)
            stdout_redir = "> /logs/STDOUT" if config.stdout_file else ""
            stderr_redir = "2> /logs/STDERR" if config.stderr_file else ""
            final_command = ["/bin/sh", "-c", f"{original_cmd} {stdout_redir} {stderr_redir}"]

        cmd.append(config.image)
        cmd.extend(final_command)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create container: {stderr.decode()}")
        return stdout.decode().strip()

    async def _start_container(self, container_id: str) -> None:
        """Start a created container."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "start",
            container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to start container: {stderr.decode()}")

    async def _wait_container(self, container_id: str, timeout: int | None) -> int:
        """Wait for container to exit."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "wait",
            container_id,
            stdout=asyncio.subprocess.PIPE,
        )

        if timeout:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        else:
            stdout, _ = await proc.communicate()

        return int(stdout.decode().strip())

    async def kill(self, container_id: str, force: bool = False) -> None:
        """Kill container."""
        signal = "SIGKILL" if force else "SIGTERM"
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "kill",
            f"--signal={signal}",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to kill container: {stderr.decode()}")

    async def remove(self, container_id: str) -> None:
        """Remove container."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to remove container: {stderr.decode()}")
