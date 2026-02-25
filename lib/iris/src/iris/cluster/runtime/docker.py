# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker runtime with cgroups v2 resource limits and BuildKit image caching.

Implements the ContainerHandle protocol with separate build and run phases:
- build(): Creates a temporary container to run setup_commands (uv sync)
- run(): Creates the main container to run the user's command

Both containers share the same workdir mount, so the .venv created during
build is available to the run container.
"""

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from iris.cluster.runtime.env import build_device_env_vars
from iris.cluster.runtime.profile import (
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    resolve_cpu_spec,
    resolve_memory_spec,
)
from iris.cluster.runtime.types import (
    ContainerConfig,
    ContainerErrorKind,
    ContainerStats,
    ContainerStatus,
    ImageInfo,
)
from iris.cluster.worker.worker_types import LogLine, TaskLogs
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)


def _build_device_flags(config: ContainerConfig) -> list[str]:
    """Build Docker device flags based on resource configuration.

    Detects TPU resources and returns appropriate Docker flags for TPU passthrough.
    Returns empty list if no special device configuration is needed.
    """
    flags: list[str] = []

    if not config.resources:
        logger.debug("No resources on container config; skipping device flags")
        return flags

    has_device = config.resources.HasField("device")
    has_tpu = has_device and config.resources.device.HasField("tpu")
    logger.info("Device flags check: has_device=%s, has_tpu=%s", has_device, has_tpu)

    if has_tpu:
        flags.extend(
            [
                "--device",
                "/dev/vfio:/dev/vfio",
                "--shm-size=100g",
                "--cap-add=SYS_RESOURCE",
                "--ulimit",
                "memlock=68719476736:68719476736",
            ]
        )
        logger.info("TPU device flags: %s", flags)

    return flags


def _detect_mount_user(mounts: list[tuple[str, str, str]]) -> str | None:
    """Detect user to run container as based on bind mount ownership.

    When bind-mounting directories owned by non-root users, the container
    must run as that user to have write access. Returns "uid:gid" for
    --user flag, or None to run as root.
    """
    for host_path, _container_path, mode in mounts:
        if "w" not in mode:
            continue
        path = Path(host_path)
        if not path.exists():
            continue
        stat = path.stat()
        if stat.st_uid != 0:
            return f"{stat.st_uid}:{stat.st_gid}"
    return None


def _parse_docker_log_line(line: str) -> tuple[datetime, str]:
    """Parse a Docker log line with timestamp prefix."""
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


def _parse_memory_size(size_str: str) -> int:
    """Parse memory size string like '123.4MiB' to MB."""
    size_str = size_str.strip()
    match = re.match(r"^([\d.]+)\s*([KMGT]i?B?)$", size_str, re.IGNORECASE)
    if not match:
        return 0

    value = float(match.group(1))
    unit = match.group(2).upper()

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


def _docker_logs(container_id: str, since: Timestamp | None = None) -> list[LogLine]:
    """Get container logs, optionally filtered by timestamp."""
    logs: list[LogLine] = []

    base_cmd = ["docker", "logs", "--timestamps"]
    if since:
        base_cmd.extend(["--since", since.as_formatted_date()])
    base_cmd.append(container_id)

    # Check if container exists
    result = subprocess.run(
        base_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    # Fetch stdout
    stdout_result = subprocess.run(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )

    for line in stdout_result.stdout.splitlines():
        if line:
            timestamp, data = _parse_docker_log_line(line)
            logs.append(LogLine(timestamp=timestamp, source="stdout", data=data))

    # Fetch stderr
    stderr_result = subprocess.run(
        base_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    for line in stderr_result.stderr.splitlines():
        if line:
            timestamp, data = _parse_docker_log_line(line)
            logs.append(LogLine(timestamp=timestamp, source="stderr", data=data))

    return logs


class DockerLogReader:
    """Incremental log reader for a Docker container using timestamp-based cursoring.

    Docker's --since flag supports sub-second precision, so advancing the cursor
    by 1ms after each read is sufficient to avoid duplicate lines.
    """

    def __init__(self, container_id: str) -> None:
        self._container_id = container_id
        self._last_timestamp: Timestamp | None = None

    def read(self) -> list[LogLine]:
        """Return new log lines since the last read. Advances the cursor by 1ms past the last line."""
        if not self._container_id:
            return []
        lines = _docker_logs(self._container_id, since=self._last_timestamp)
        if lines:
            max_ts = max(line.timestamp for line in lines)
            self._last_timestamp = Timestamp.from_seconds(max_ts.timestamp()).add_ms(1)
        return lines

    def read_all(self) -> list[LogLine]:
        """Return all logs from the beginning."""
        if not self._container_id:
            return []
        return _docker_logs(self._container_id)


@dataclass
class DockerContainerHandle:
    """Docker implementation of ContainerHandle.

    Implements a two-phase execution model:
    - build(): Run setup_commands in a temporary container to create .venv
    - run(): Run the main command in a container that uses the created .venv

    Both containers share the same workdir mount (/app), so the .venv created
    during build is available to the run container. This separation enables
    scheduler back-pressure on the BUILDING phase.
    """

    config: ContainerConfig
    runtime: "DockerRuntime"
    _run_container_id: str | None = field(default=None, repr=False)

    @property
    def container_id(self) -> str | None:
        """Return the current run container ID, if any."""
        return self._run_container_id

    def build(self) -> list[LogLine]:
        """Run setup_commands (uv sync, pip install, etc) in a temporary container.

        Creates a temporary container that runs setup_commands, waits for completion,
        and removes the container. The .venv is created in the shared workdir mount
        and will be available to the run container.

        If there are no setup_commands, this is a no-op.

        Returns:
            List of log lines captured during the build phase.

        Raises:
            RuntimeError: If setup fails (non-zero exit code)
        """
        if not self.config.entrypoint.setup_commands:
            logger.debug("No setup_commands, skipping build phase")
            return []

        # Build a bash script that runs all setup commands
        setup_script = self._generate_setup_script()
        self._write_setup_script(setup_script)

        # Create and run the build container
        build_container_id = self._docker_create(
            command=["bash", "/app/_setup_env.sh"],
            label_suffix="_build",
        )

        build_logs: list[LogLine] = []
        try:
            self._docker_start(build_container_id)

            # Wait for build to complete (blocking), streaming logs incrementally
            last_log_time: Timestamp | None = None
            while True:
                status = self._docker_inspect(build_container_id)

                # Capture logs incrementally during build
                new_logs = _docker_logs(build_container_id, since=last_log_time)
                if new_logs:
                    build_logs.extend(new_logs)
                    last_log_time = Timestamp.from_seconds(new_logs[-1].timestamp.timestamp()).add_ms(1)

                if not status.running:
                    break
                time.sleep(0.5)

            # Final log fetch after container stops
            final_logs = _docker_logs(build_container_id, since=last_log_time)
            build_logs.extend(final_logs)

            if status.exit_code != 0:
                log_text = "\n".join(f"[{entry.source}] {entry.data}" for entry in build_logs[-50:])
                raise RuntimeError(f"Build failed with exit_code={status.exit_code}\n" f"Last 50 log lines:\n{log_text}")

            logger.info("Build phase completed successfully for task %s", self.config.task_id)
            return build_logs

        finally:
            # Always clean up the build container
            self._docker_remove(build_container_id)

    def _generate_setup_script(self) -> str:
        """Generate a bash script that runs setup commands."""
        lines = ["#!/bin/bash", "set -e"]
        lines.extend(self.config.entrypoint.setup_commands)
        return "\n".join(lines) + "\n"

    def _write_setup_script(self, script: str) -> None:
        """Write the setup script to the workdir mount."""
        for host_path, container_path, _mode in self.config.mounts:
            if container_path == "/app":
                (Path(host_path) / "_setup_env.sh").write_text(script)
                return
        raise RuntimeError("No /app mount found in config")

    def run(self) -> None:
        """Start the main command container.

        Non-blocking - returns immediately after starting the container.
        Use status() to monitor execution progress.
        """
        # Build the run command: activate venv then exec user command
        quoted_cmd = " ".join(shlex.quote(arg) for arg in self.config.entrypoint.run_command.argv)

        # If we had setup_commands, the venv exists and we should activate it
        if self.config.entrypoint.setup_commands:
            run_script = f"""#!/bin/bash
set -e
cd /app
source .venv/bin/activate
exec {quoted_cmd}
"""
            self._write_run_script(run_script)
            command = ["bash", "/app/_run.sh"]
        else:
            # No setup, run command directly
            command = list(self.config.entrypoint.run_command.argv)

        self._run_container_id = self._docker_create(
            command=command,
            include_resources=True,
        )
        self.runtime._track_container(self._run_container_id)
        self._docker_start(self._run_container_id)

        logger.info(
            "Run phase started for task %s (container_id=%s)",
            self.config.task_id,
            self._run_container_id,
        )

    def _write_run_script(self, script: str) -> None:
        """Write the run script to the workdir mount."""
        for host_path, container_path, _mode in self.config.mounts:
            if container_path == "/app":
                (Path(host_path) / "_run.sh").write_text(script)
                return
        raise RuntimeError("No /app mount found in config")

    def stop(self, force: bool = False) -> None:
        """Stop the run container."""
        if self._run_container_id:
            self._docker_kill(self._run_container_id, force)

    def status(self) -> ContainerStatus:
        """Check container status (running, exit code, error)."""
        if not self._run_container_id:
            return ContainerStatus(running=False, error="Container not started")
        return self._docker_inspect(self._run_container_id)

    def log_reader(self) -> DockerLogReader:
        """Create an incremental log reader for this container."""
        return DockerLogReader(self._run_container_id or "")

    def stats(self) -> ContainerStats:
        """Get resource usage statistics."""
        if not self._run_container_id:
            return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)
        return self._docker_stats(self._run_container_id)

    def profile(self, duration_seconds: int, profile_type: "cluster_pb2.ProfileType") -> bytes:
        """Profile the running process using py-spy (CPU) or memray (memory)."""

        container_id = self._run_container_id
        if not container_id:
            raise RuntimeError("Cannot profile: no running container")

        profile_id = uuid.uuid4().hex[:8]

        # Dispatch to CPU or memory profiling based on profile_type
        if profile_type.HasField("cpu"):
            return self._profile_cpu(container_id, duration_seconds, profile_type.cpu, profile_id)
        elif profile_type.HasField("memory"):
            return self._profile_memory(container_id, duration_seconds, profile_type.memory, profile_id)
        else:
            raise RuntimeError("ProfileType must specify either cpu or memory profiler")

    def _docker_exec(self, container_id: str, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", "exec", container_id, *cmd], **kwargs)

    def _docker_read_file(self, container_id: str, path: str) -> bytes:
        result = self._docker_exec(container_id, ["cat", path], capture_output=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to read {path}: {result.stderr}")
        return result.stdout

    def _docker_rm_files(self, container_id: str, paths: list[str]) -> None:
        self._docker_exec(container_id, ["rm", "-f", *paths], capture_output=True, timeout=10)

    def _profile_cpu(
        self, container_id: str, duration_seconds: int, cpu_config: "cluster_pb2.CpuProfile", profile_id: str
    ) -> bytes:
        """Profile CPU using py-spy."""
        spec = resolve_cpu_spec(cpu_config, duration_seconds, pid="1")
        output_path = f"/tmp/profile-cpu-{profile_id}.{spec.ext}"
        cmd = build_pyspy_cmd(spec, py_spy_bin="/app/.venv/bin/py-spy", output_path=output_path)

        logger.info(
            "CPU profiling container %s for %ds (format=%s, rate=%dHz)",
            container_id,
            duration_seconds,
            spec.py_spy_format,
            spec.rate_hz,
        )
        try:
            result = self._docker_exec(container_id, cmd, capture_output=True, text=True, timeout=duration_seconds + 5)
            if result.returncode != 0:
                raise RuntimeError(f"py-spy failed: {result.stderr}")
            return self._docker_read_file(container_id, output_path)
        finally:
            self._docker_rm_files(container_id, [output_path])

    def _profile_memory(
        self, container_id: str, duration_seconds: int, memory_config: "cluster_pb2.MemoryProfile", profile_id: str
    ) -> bytes:
        """Profile memory using memray."""
        spec = resolve_memory_spec(memory_config, duration_seconds, pid="1")
        memray_bin = "/app/.venv/bin/memray"
        trace_path = f"/tmp/memray-trace-{profile_id}.bin"
        output_path = f"/tmp/memray-output-{profile_id}.{spec.ext}"

        attach_cmd = build_memray_attach_cmd(spec, memray_bin, trace_path)
        transform_cmd = build_memray_transform_cmd(spec, memray_bin, trace_path, output_path)

        logger.info(
            "Memory profiling container %s for %ds (format=%s, leaks=%s)",
            container_id,
            duration_seconds,
            spec.reporter,
            spec.leaks,
        )
        try:
            result = self._docker_exec(
                container_id, attach_cmd, capture_output=True, text=True, timeout=duration_seconds + 10
            )
            if result.returncode != 0:
                raise RuntimeError(f"memray attach failed: {result.stderr}")

            result = self._docker_exec(container_id, transform_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"memray {spec.reporter} failed: {result.stderr}")

            if spec.output_is_file:
                return self._docker_read_file(container_id, output_path)
            else:
                return result.stdout.encode("utf-8")
        finally:
            self._docker_rm_files(container_id, [trace_path, output_path])

    def cleanup(self) -> None:
        """Remove the run container and clean up resources."""
        if self._run_container_id:
            self._docker_remove(self._run_container_id)
            self.runtime._untrack_container(self._run_container_id)
            self._run_container_id = None

    # -------------------------------------------------------------------------
    # Docker CLI helpers
    # -------------------------------------------------------------------------

    def _docker_create(
        self,
        command: list[str],
        label_suffix: str = "",
        include_resources: bool = False,
    ) -> str:
        """Create a Docker container. Returns container_id."""
        config = self.config

        cmd = [
            "docker",
            "create",
            "--security-opt",
            "no-new-privileges",
            "-w",
            config.workdir,
        ]

        # Run as the owner of bind-mounted directories
        user_flag = _detect_mount_user(config.mounts)
        if user_flag:
            cmd.extend(["--user", user_flag])

        if config.network_mode:
            cmd.extend(["--network", config.network_mode])
        else:
            cmd.append("--add-host=host.docker.internal:host-gateway")

        cmd.extend(["--cap-drop", "ALL"])
        cmd.extend(["--cap-add", "SYS_PTRACE"])

        # Device flags (TPU passthrough etc) - only for run container
        if include_resources:
            cmd.extend(_build_device_flags(config))

        # Labels for discoverability
        cmd.extend(["--label", "iris.managed=true"])
        if config.task_id:
            cmd.extend(["--label", f"iris.task_id={config.task_id}{label_suffix}"])
        if config.job_id:
            cmd.extend(["--label", f"iris.job_id={config.job_id}"])

        # Resource limits (cgroups v2) - only for run container
        if include_resources:
            cpu_millicores = config.get_cpu_millicores()
            if cpu_millicores:
                cpus = cpu_millicores / 1000
                cmd.extend(["--cpus", str(cpus)])
            memory_mb = config.get_memory_mb()
            if memory_mb:
                cmd.extend(["--memory", f"{memory_mb}m"])

        # Build combined environment
        device_env = build_device_env_vars(config) if include_resources else {}
        combined_env = {**device_env, **config.env}

        for k, v in combined_env.items():
            cmd.extend(["-e", f"{k}={v}"])

        # Mounts
        for host, container, mode in config.mounts:
            cmd.extend(["-v", f"{host}:{container}:{mode}"])

        cmd.append(config.image)
        cmd.extend(command)

        logger.info("Creating container: %s", " ".join(cmd[:20]))
        logger.debug("Full docker create command: %s", cmd)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")

        return result.stdout.strip()

    def _docker_start(self, container_id: str) -> None:
        """Start a Docker container."""
        result = subprocess.run(
            ["docker", "start", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

    def _docker_inspect(self, container_id: str) -> ContainerStatus:
        """Inspect container status."""
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
            return ContainerStatus(
                running=False,
                error=f"Container not found: id={container_id}",
                error_kind=ContainerErrorKind.INFRA_NOT_FOUND,
            )

        try:
            state = json.loads(result.stdout.strip())
            running = state.get("Running", False)
            exit_code = state.get("ExitCode")
            error_msg = state.get("Error", "") or None
            oom_killed = state.get("OOMKilled", False)

            return ContainerStatus(
                running=running,
                exit_code=exit_code if not running else None,
                error=error_msg,
                error_kind=ContainerErrorKind.USER_CODE if error_msg else ContainerErrorKind.NONE,
                oom_killed=oom_killed,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return ContainerStatus(
                running=False,
                error=f"Failed to parse inspect output: {e}",
                error_kind=ContainerErrorKind.RUNTIME_ERROR,
            )

    def _docker_stats(self, container_id: str) -> ContainerStats:
        """Get container stats."""
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

            memory_str = stats.get("MemUsage", "0B / 0B").split("/")[0].strip()
            memory_mb = _parse_memory_size(memory_str)

            cpu_str = stats.get("CPUPerc", "0%").rstrip("%")
            cpu_percent = int(float(cpu_str)) if cpu_str else 0

            pids_str = stats.get("PIDs", "0")
            process_count = int(pids_str) if pids_str.isdigit() else 0

            return ContainerStats(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                process_count=process_count,
                available=True,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return ContainerStats(
                memory_mb=0,
                cpu_percent=0,
                process_count=0,
                available=False,
            )

    def _docker_kill(self, container_id: str, force: bool = False) -> None:
        """Kill container."""
        status = self._docker_inspect(container_id)
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

    def _docker_remove(self, container_id: str) -> None:
        """Remove container."""
        result = subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("Failed to remove container %s: %s", container_id, result.stderr)


class DockerRuntime:
    """Runtime that creates DockerContainerHandle instances.

    Tracks all created containers for cleanup on shutdown.
    """

    def __init__(self) -> None:
        self._handles: list[DockerContainerHandle] = []
        self._created_containers: set[str] = set()

    def create_container(self, config: ContainerConfig) -> DockerContainerHandle:
        """Create a container handle from config.

        The handle is not started - call handle.build() then handle.run()
        to execute the container.
        """
        handle = DockerContainerHandle(config=config, runtime=self)
        self._handles.append(handle)
        return handle

    def stage_bundle(
        self,
        *,
        bundle_gcs_path: str,
        workdir: Path,
        workdir_files: dict[str, bytes],
        fetch_bundle: Callable[[str], Path],
    ) -> None:
        """Stage bundle and workdir files on worker-local filesystem."""
        bundle_path = fetch_bundle(bundle_gcs_path)
        shutil.copytree(bundle_path, workdir, dirs_exist_ok=True)
        for name, data in workdir_files.items():
            (workdir / name).write_bytes(data)

    def _track_container(self, container_id: str) -> None:
        """Track a container ID for cleanup."""
        self._created_containers.add(container_id)

    def _untrack_container(self, container_id: str) -> None:
        """Untrack a container ID."""
        self._created_containers.discard(container_id)

    def list_containers(self) -> list[DockerContainerHandle]:
        """List all managed container handles."""
        return list(self._handles)

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

    def cleanup(self) -> None:
        """Clean up all containers managed by this runtime."""
        for handle in self._handles:
            handle.cleanup()
        self._handles.clear()

        # Also clean up any containers that weren't cleaned up via handles
        for cid in list(self._created_containers):
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)
        self._created_containers.clear()


class DockerImageBuilder:
    """Build Docker images using Docker CLI with BuildKit."""

    def __init__(self) -> None:
        pass

    def build(
        self,
        dockerfile_content: str,
        tag: str,
        task_logs: TaskLogs | None = None,
        context: Path | None = None,
    ) -> None:
        """Build a Docker image using context as build context directory."""
        if context is None:
            raise ValueError("context (bundle_path) is required for Docker builds")
        dockerfile_path = context / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        context_dir = str(context)

        if task_logs:
            task_logs.add("build", f"Starting build for image: {tag}")

        cmd = [
            "docker",
            "build",
            "-t",
            tag,
            context_dir,
        ]

        proc = subprocess.Popen(
            cmd,
            env={**os.environ, "DOCKER_BUILDKIT": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

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
