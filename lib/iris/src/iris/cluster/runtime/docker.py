# Copyright The Marin Authors
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
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rigging.timing import Timestamp

from iris.cluster.bundle import BundleStore
from iris.cluster.log_keys import STDERR_SOURCE, STDOUT_SOURCE
from iris.cluster.runtime.env import VENV_PATH, render_setup_steps, write_workdir_files
from iris.cluster.runtime.profile import (
    PROFILER_WATCHDOG_GRACE_SECONDS,
    ExecResult,
    capture_cpu,
    capture_memory_attach,
    capture_threads,
    sigcont_sweep_argv,
    wrap_with_kill_watchdog,
)
from iris.cluster.runtime.types import (
    ContainerConfig,
    ContainerErrorKind,
    ContainerInfraError,
    ContainerPhase,
    ContainerStats,
    ContainerStatus,
    DiscoveredContainer,
    ExecutionStage,
    MountKind,
    MountSpec,
)
from iris.cluster.types import CapacityType
from iris.cluster.worker.worker_types import LogLine, TaskLogs
from iris.rpc import job_pb2
from iris.rpc.proto_display import resolve_container_profile

logger = logging.getLogger(__name__)

# Substrings that indicate a docker/registry infrastructure problem rather than
# a user-code error.  Checked case-insensitively against stderr from docker
# create/start/pull.
_INFRA_ERROR_PATTERNS: list[str] = [
    "error getting credentials",
    "denied: denied",
    "unauthorized: authentication required",
    "connection refused",
    "dial tcp",
    "no such host",
    "i/o timeout",
    "TLS handshake timeout",
    "daemon is not running",
    "Cannot connect to the Docker daemon",
]


def _is_docker_infra_error(stderr: str) -> bool:
    """Return True if *stderr* matches a known infrastructure failure pattern."""
    stderr_lower = stderr.lower()
    return any(p.lower() in stderr_lower for p in _INFRA_ERROR_PATTERNS)


@dataclass(frozen=True)
class _DockerProfileDispatch:
    """``ProfileDispatch`` backed by ``docker exec`` into a task container.

    Profilers run in the container's isolated PID namespace, so ``exec_profiler``
    applies the kill-watchdog and SIGCONT recovery (see ``runtime.profile``);
    ``exec`` runs non-attaching transform/read commands directly.
    """

    container_id: str
    pyspy_bin: str = f"{VENV_PATH}/bin/py-spy"
    memray_bin: str = f"{VENV_PATH}/bin/memray"

    @contextmanager
    def scratch(self, *suffixes: str) -> Iterator[tuple[str, ...]]:
        paths = tuple(f"/tmp/iris-profile-{uuid.uuid4().hex[:8]}.{suffix}" for suffix in suffixes)
        try:
            yield paths
        finally:
            subprocess.run(["docker", "exec", self.container_id, "rm", "-f", *paths], capture_output=True, timeout=10)

    def exec_profiler(self, cmd: list[str], *, sample_timeout: int) -> ExecResult:
        watchdog_cmd = wrap_with_kill_watchdog(cmd, sample_timeout)
        try:
            return self.exec(watchdog_cmd, timeout=sample_timeout + PROFILER_WATCHDOG_GRACE_SECONDS)
        finally:
            self._sigcont_sweep()

    def exec(self, cmd: list[str], *, timeout: int) -> ExecResult:
        result = subprocess.run(
            ["docker", "exec", self.container_id, *cmd], capture_output=True, text=True, timeout=timeout
        )
        return ExecResult(result.returncode, (result.stdout or "").encode("utf-8"), result.stderr or "")

    def read_file(self, path: str) -> bytes:
        result = subprocess.run(["docker", "exec", self.container_id, "cat", path], capture_output=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to read {path}: {result.stderr.decode('utf-8', 'replace')}")
        return result.stdout

    def _sigcont_sweep(self) -> None:
        try:
            subprocess.run(
                ["docker", "exec", self.container_id, *sigcont_sweep_argv()],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("SIGCONT sweep failed for container %s: %s", self.container_id, e)


def _resolve_profiler_bin(container_id: str, venv_bin: str, fallback: str) -> str:
    """Prefer the venv-installed profiler, falling back to PATH for BYO images.

    iris installs py-spy/memray into ``$IRIS_VENV``, but a bring-your-own image
    with no venv may carry them on PATH instead.
    """
    probe = subprocess.run(["docker", "exec", container_id, "test", "-x", venv_bin], capture_output=True, timeout=5)
    return venv_bin if probe.returncode == 0 else fallback


# Widened ephemeral port range for high-connection workloads: the controller and
# workers fan out to thousands of peers and would otherwise exhaust the default
# ~28k-port pool. The floor is deliberately above every fixed port the cluster's
# own services bind — the TPU/JAX runtime (8081, 8431, 8470-8482) and iris'
# controller/worker RPC (10000/10001). An earlier floor of 1024 pulled those into
# the kernel's random ephemeral pool, so a co-tenant's outbound connection could
# be handed e.g. 8431 and block the TPU trainer from binding it, crash-looping
# with "[::]:8431 ... Address already in use". 11000-65535 still leaves ~54k
# ephemeral ports.
EPHEMERAL_PORT_RANGE = "11000 65535"

# Belt-and-suspenders for the fixed TPU/JAX ports that sit just below the floor:
# reserve them so they stay out of automatic ephemeral assignment even if the
# floor is ever lowered again. An explicit bind() by the TPU runtime is
# unaffected. libtpu's Runtime Metric Service is 8431; 8470-8482 is the Cloud TPU
# runtime/SliceBuilder block (incl. the JAX coordinator 8482 and marin's default
# 8476); 8081 is levanter megascale.
RESERVED_HOST_PORTS = "8081,8431,8470-8482"

# Network sysctl tuning for containers with their own network namespace (#3066).
# Host-network containers inherit host settings (configured at VM bootstrap).
_NETWORK_SYSCTLS: dict[str, str] = {
    "net.ipv4.ip_local_port_range": EPHEMERAL_PORT_RANGE,
    "net.ipv4.ip_local_reserved_ports": RESERVED_HOST_PORTS,
    "net.ipv4.tcp_tw_reuse": "1",
}


@dataclass(frozen=True)
class ResolvedMount:
    """A MountSpec resolved to concrete host and container paths for Docker."""

    host_path: str
    container_path: str
    mode: str  # "rw" or "ro"
    kind: MountKind


def _has_tpu_device(config: ContainerConfig) -> bool:
    """Return True when config requests TPU resources."""
    if not config.resources:
        return False
    has_device = config.resources.HasField("device")
    return has_device and config.resources.device.HasField("tpu")


def _build_device_flags(config: ContainerConfig) -> list[str]:
    """Build Docker device flags based on resource configuration.

    Detects TPU resources and returns the Docker flags for TPU passthrough: large
    shared memory, the locked-memory ulimit, and the SYS_RESOURCE capability for
    memlock. Privilege (``--privileged``) is handled by the security-profile flags,
    not here. Returns an empty list when no special device configuration is needed.
    """
    flags: list[str] = []

    if not config.resources:
        logger.debug("No resources on container config; skipping device flags")
        return flags

    has_device = config.resources.HasField("device")
    has_tpu = _has_tpu_device(config)
    logger.info("Device flags check: has_device=%s, has_tpu=%s", has_device, has_tpu)

    if has_tpu:
        flags.extend(
            [
                "--shm-size=100g",
                "--cap-add=SYS_RESOURCE",
                "--ulimit",
                "memlock=68719476736:68719476736",
            ]
        )
        logger.info("TPU device flags: %s", flags)

    return flags


def _security_flags(profile: int, is_tpu_run: bool) -> list[str]:
    """Docker security flags (privilege, capabilities, docker-socket mount).

    A TPU run requires ``--privileged`` for device access regardless of profile,
    so on TPU even RESTRICTED/DEFAULT run privileged.
    """
    resolved = resolve_container_profile(profile)
    privileged = resolved == job_pb2.CONTAINER_PROFILE_PRIVILEGED or is_tpu_run

    flags: list[str] = []
    if privileged:
        flags.append("--privileged")
    else:
        flags.extend(["--security-opt", "no-new-privileges", "--cap-drop", "ALL"])

    # SYS_PTRACE lets py-spy attach via `docker exec`. RESTRICTED deliberately
    # omits it; every other profile adds it back (a privileged container nominally
    # has it, but exec'd processes don't reliably inherit it).
    if resolved != job_pb2.CONTAINER_PROFILE_RESTRICTED:
        flags.extend(["--cap-add", "SYS_PTRACE"])

    if resolved == job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS:
        flags.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])

    return flags


def _detect_mount_user(mounts: list[ResolvedMount]) -> str | None:
    """Detect user to run container as based on bind mount ownership.

    When bind-mounting directories owned by non-root users, the container
    must run as that user to have write access. Returns "uid:gid" for
    --user flag, or None to run as root.
    """
    for mount in mounts:
        if "w" not in mount.mode:
            continue
        path = Path(mount.host_path)
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
    return datetime.now(UTC), line


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
    """Get container logs, optionally filtered by timestamp.

    Uses a single `docker logs` call with capture_output to get both stdout and
    stderr in one shot, then parses each stream separately.
    """
    cmd = ["docker", "logs", "--timestamps"]
    if since:
        cmd.extend(["--since", since.as_formatted_date()])
    cmd.append(container_id)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []

    logs: list[LogLine] = []
    for line in result.stdout.splitlines():
        if line:
            timestamp, data = _parse_docker_log_line(line)
            logs.append(LogLine(timestamp=timestamp, source=STDOUT_SOURCE, data=data))
    for line in result.stderr.splitlines():
        if line:
            timestamp, data = _parse_docker_log_line(line)
            logs.append(LogLine(timestamp=timestamp, source=STDERR_SOURCE, data=data))
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
    _resolved_mounts: list[ResolvedMount] = field(default_factory=list, repr=False)
    _run_container_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_existing(cls, container_id: str, runtime: "DockerRuntime") -> "DockerContainerHandle":
        """Wrap an already-running container for adoption after worker restart.

        Skips container creation — the container already exists.
        The returned handle supports status(), stop(), log_reader(), stats(),
        and cleanup(), but build()/run() should not be called.
        """
        # Minimal config — we only need it for status/stop/log operations
        # which don't reference config fields.
        config = ContainerConfig(
            image="",
            entrypoint=job_pb2.RuntimeEntrypoint(),
            env={},
        )
        handle = cls(config=config, runtime=runtime, _run_container_id=container_id)
        runtime.track_container(container_id)
        return handle

    @property
    def container_id(self) -> str | None:
        """Return the Docker container ID (hash), if any."""
        return self._run_container_id

    def build(self, on_logs: Callable[[list[LogLine]], None] | None = None) -> list[LogLine]:
        """Run setup_commands (uv sync, pip install, etc) in a temporary container.

        Creates a temporary container that runs setup_commands, waits for completion,
        and removes the container. The .venv is created in the shared workdir mount
        and will be available to the run container.

        If there are no setup_commands, this is a no-op.

        Args:
            on_logs: Optional callback invoked with each incremental batch of
                log lines during the build. Enables streaming to LogService.

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

        # Build containers get max(32 GB, task request) memory — uv sync on a large
        # workspace OOMed at the old 8 GB ceiling on a host with 1.4 TB free.
        task_memory_bytes = self.config.resources.memory_bytes if self.config.resources else 0
        build_memory_bytes = (
            max(self._BUILD_MEMORY_LIMIT_BYTES, task_memory_bytes)
            if task_memory_bytes
            else self._BUILD_MEMORY_LIMIT_BYTES
        )
        build_memory_mb = build_memory_bytes // (1024 * 1024)

        build_container_id = self._docker_create(
            command=["bash", "/app/_setup_env.sh"],
            label_suffix="_build",
            memory_limit_mb=build_memory_mb,
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
                    if on_logs:
                        on_logs(new_logs)

                if status.phase == ContainerPhase.STOPPED:
                    break
                time.sleep(0.5)

            # Final log fetch after container stops
            final_logs = _docker_logs(build_container_id, since=last_log_time)
            if final_logs:
                build_logs.extend(final_logs)
                if on_logs:
                    on_logs(final_logs)

            if status.exit_code != 0:
                log_text = "\n".join(f"[{entry.source}] {entry.data}" for entry in build_logs[-50:])
                raise RuntimeError(f"Build failed with exit_code={status.exit_code}\nLast 50 log lines:\n{log_text}")

            logger.info("Build phase completed successfully for task %s", self.config.task_id)
            return build_logs

        finally:
            # Always clean up the build container
            self._docker_remove(build_container_id)

    def _generate_setup_script(self) -> str:
        """Generate a bash script that runs each setup command as its own step."""
        lines = ["#!/bin/bash", "set -e"]
        lines.extend(render_setup_steps(self.config.entrypoint.setup_commands))
        return "\n".join(lines) + "\n"

    def _write_setup_script(self, script: str) -> None:
        """Write the setup script to the workdir mount."""
        for rm in self._resolved_mounts:
            if rm.container_path == "/app":
                (Path(rm.host_path) / "_setup_env.sh").write_text(script)
                return
        raise RuntimeError("No /app mount found in config")

    def run(self) -> None:
        """Start the main command container.

        Non-blocking - returns immediately after starting the container.
        Use status() to monitor execution progress.
        """
        quoted_cmd = " ".join(shlex.quote(arg) for arg in self.config.entrypoint.run_command.argv)

        # Run from the workdir (matching the k8s task script) and activate the venv
        # only when a setup script left one, so a bring-your-own-env command runs in
        # the image as-is instead of failing on a missing .venv.
        run_script = f"""#!/bin/bash
set -e
cd "$IRIS_WORKDIR"
[ -f "$IRIS_VENV/bin/activate" ] && source "$IRIS_VENV/bin/activate"
exec {quoted_cmd}
"""
        self._write_run_script(run_script)
        command = ["bash", "/app/_run.sh"]

        self._run_container_id = self._docker_create(
            command=command,
            include_devices=True,
        )
        self.runtime.track_container(self._run_container_id)
        self._docker_start(self._run_container_id)

        logger.info(
            "Run phase started for task %s (container_id=%s)",
            self.config.task_id,
            self._run_container_id,
        )

    def _write_run_script(self, script: str) -> None:
        """Write the run script to the workdir mount."""
        for rm in self._resolved_mounts:
            if rm.container_path == "/app":
                (Path(rm.host_path) / "_run.sh").write_text(script)
                return
        raise RuntimeError("No /app mount found in config")

    def stop(self, force: bool = False) -> None:
        """Stop the run container."""
        if self._run_container_id:
            self._docker_kill(self._run_container_id, force)

    def status(self) -> ContainerStatus:
        """Check container status (running, exit code, error)."""
        if not self._run_container_id:
            return ContainerStatus(phase=ContainerPhase.STOPPED, error="Container not started")
        return self._docker_inspect(self._run_container_id)

    def log_reader(self) -> DockerLogReader:
        """Create an incremental log reader for this container."""
        return DockerLogReader(self._run_container_id or "")

    def stats(self) -> ContainerStats:
        """Get resource usage statistics."""
        if not self._run_container_id:
            return ContainerStats(memory_mb=0, cpu_millicores=0, process_count=0, available=False)
        return self._docker_stats(self._run_container_id)

    def disk_usage_mb(self) -> int:
        """Return used space in MB on the filesystem containing the workdir."""
        for rm in self._resolved_mounts:
            if rm.container_path == self.config.workdir:
                path = Path(rm.host_path)
                if path.exists():
                    return int(shutil.disk_usage(path).used / (1024 * 1024))
        return 0

    def profile(self, duration_seconds: int, profile_type: "job_pb2.ProfileType") -> bytes:
        """Profile the running process using py-spy (CPU), memray (memory), or thread dump."""
        container_id = self._run_container_id
        if not container_id:
            raise RuntimeError("Cannot profile: no running container")

        dispatch = _DockerProfileDispatch(
            container_id,
            pyspy_bin=_resolve_profiler_bin(container_id, f"{VENV_PATH}/bin/py-spy", "py-spy"),
            memray_bin=_resolve_profiler_bin(container_id, f"{VENV_PATH}/bin/memray", "memray"),
        )
        if profile_type.HasField("threads"):
            return capture_threads(dispatch, pid="1", include_locals=profile_type.threads.locals)
        elif profile_type.HasField("cpu"):
            return capture_cpu(dispatch, profile_type.cpu, duration_seconds, pid="1")
        elif profile_type.HasField("memory"):
            return capture_memory_attach(dispatch, profile_type.memory, duration_seconds, pid="1")
        else:
            raise RuntimeError("ProfileType must specify cpu, memory, or threads profiler")

    def cleanup(self) -> None:
        """Remove the run container and clean up resources."""
        if self._run_container_id:
            self._docker_remove(self._run_container_id)
            self.runtime.untrack_container(self._run_container_id)
            self._run_container_id = None

    # -------------------------------------------------------------------------
    # Docker CLI helpers
    # -------------------------------------------------------------------------

    _BUILD_MEMORY_LIMIT_BYTES = 32 * 1024**3

    def _docker_create(
        self,
        command: list[str],
        label_suffix: str = "",
        include_devices: bool = False,
        memory_limit_mb: int | None = None,
    ) -> str:
        """Create a Docker container. Returns container_id.

        CPU and memory cgroup limits are always applied. Device passthrough
        (TPU/GPU) is only enabled for run containers via include_devices.

        Args:
            command: Command to run in the container.
            label_suffix: Suffix appended to the iris.task_id label.
            include_devices: If True, also pass through accelerator devices.
            memory_limit_mb: Override memory limit in MB. When None, uses
                the task's requested memory from config.resources.
        """
        config = self.config
        self.runtime.ensure_image(config.image)

        cmd = [
            "docker",
            "create",
            "--ulimit",
            "core=0:0",
            "--ulimit",
            "nofile=65536:524288",
            "-w",
            config.workdir,
        ]
        is_tpu_run = include_devices and _has_tpu_device(config)

        cmd.extend(_security_flags(config.container_profile, is_tpu_run))
        logger.info(
            "Container security profile %s (tpu_run=%s) for task %s",
            job_pb2.ContainerProfile.Name(resolve_container_profile(config.container_profile)),
            is_tpu_run,
            config.task_id,
        )

        # Run as the owner of bind-mounted directories
        user_flag = _detect_mount_user(self._resolved_mounts)
        if user_flag:
            cmd.extend(["--user", user_flag])

        if config.network_mode:
            cmd.extend(["--network", config.network_mode])
        else:
            cmd.append("--add-host=host.docker.internal:host-gateway")

        # Network sysctl tuning for containers with own network namespace (#3066).
        # Host-network containers inherit host settings from VM bootstrap.
        if config.network_mode != "host":
            for key, value in _NETWORK_SYSCTLS.items():
                cmd.extend(["--sysctl", f"{key}={value}"])

        # Device flags (TPU passthrough etc) - only for run container
        if include_devices:
            cmd.extend(_build_device_flags(config))

        # Labels for discoverability and container adoption after worker restart
        cmd.extend(["--label", "iris.managed=true"])
        if config.task_id:
            cmd.extend(["--label", f"iris.task_id={config.task_id}{label_suffix}"])
        if config.job_id:
            cmd.extend(["--label", f"iris.job_id={config.job_id}"])
        if config.attempt_id is not None:
            cmd.extend(["--label", f"iris.attempt_id={config.attempt_id}"])
        if config.attempt_uid:
            cmd.extend(["--label", f"iris.attempt_uid={config.attempt_uid}"])
        if config.worker_id:
            cmd.extend(["--label", f"iris.worker_id={config.worker_id}"])
        # Phase label: used during adoption to distinguish adoptable run
        # containers from transient build containers that should be cleaned up.
        phase = ExecutionStage.BUILD if label_suffix == "_build" else ExecutionStage.RUN
        cmd.extend(["--label", f"iris.phase={phase}"])
        # Host-port reservations, so a restarted worker can re-mark them as
        # taken when it adopts this container (otherwise the ports are dropped
        # and can be double-allocated to a new task).
        if config.ports:
            cmd.extend(["--label", f"iris.ports={json.dumps(config.ports)}"])

        # Resource limits (cgroups v2) — always applied
        cpu_millicores = config.get_cpu_millicores()
        if cpu_millicores:
            if self.runtime.capacity_type == CapacityType.ON_DEMAND:
                # Soft weight: on-demand workers let containers burst onto idle
                # host CPU; the scheduler still places by cpu_millicores.
                shares = max(2, int(cpu_millicores * 1024 / 1000))
                cmd.extend(["--cpu-shares", str(shares)])
            else:
                cmd.extend(["--cpus", str(cpu_millicores / 1000)])
        effective_memory_mb = memory_limit_mb or config.get_memory_mb()
        if effective_memory_mb:
            cmd.extend(["--memory", f"{effective_memory_mb}m"])

        # Device env vars (TPU/GPU) are now included in config.env by
        # build_common_iris_env(), so no separate device_env merge needed.
        combined_env = dict(config.env)

        for k, v in combined_env.items():
            cmd.extend(["-e", f"{k}={v}"])

        # Mounts
        for rm in self._resolved_mounts:
            if rm.kind == MountKind.TMPFS:
                # Use Docker --tmpfs for per-container isolation instead of shared bind mount
                cmd.extend(["--tmpfs", rm.container_path])
            else:
                cmd.extend(["-v", f"{rm.host_path}:{rm.container_path}:{rm.mode}"])

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
            stderr = result.stderr
            if _is_docker_infra_error(stderr):
                raise ContainerInfraError(f"Failed to create container (infra): {stderr}")
            raise RuntimeError(f"Failed to create container: {stderr}")

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
            stderr = result.stderr
            if _is_docker_infra_error(stderr):
                raise ContainerInfraError(f"Failed to start container (infra): {stderr}")
            raise RuntimeError(f"Failed to start container: {stderr}")

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
                phase=ContainerPhase.STOPPED,
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
                phase=ContainerPhase.RUNNING if running else ContainerPhase.STOPPED,
                exit_code=exit_code if not running else None,
                error=error_msg,
                error_kind=ContainerErrorKind.USER_CODE if error_msg else ContainerErrorKind.NONE,
                oom_killed=oom_killed,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return ContainerStatus(
                phase=ContainerPhase.STOPPED,
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
                cpu_millicores=0,
                process_count=0,
                available=False,
            )

        try:
            stats = json.loads(result.stdout.strip())

            memory_str = stats.get("MemUsage", "0B / 0B").split("/")[0].strip()
            memory_mb = _parse_memory_size(memory_str)

            cpu_str = stats.get("CPUPerc", "0%").rstrip("%")
            # Docker reports CPUPerc with 100% == one fully utilized CPU core, so
            # converting to millicores is a straight percent * 10. See
            # https://docs.docker.com/reference/cli/docker/container/stats/
            cpu_millicores = int(float(cpu_str) * 10) if cpu_str else 0

            pids_str = stats.get("PIDs", "0")
            process_count = int(pids_str) if pids_str.isdigit() else 0

            return ContainerStats(
                memory_mb=memory_mb,
                cpu_millicores=cpu_millicores,
                process_count=process_count,
                available=True,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return ContainerStats(
                memory_mb=0,
                cpu_millicores=0,
                process_count=0,
                available=False,
            )

    def _docker_kill(self, container_id: str, force: bool = False) -> None:
        """Kill container."""
        status = self._docker_inspect(container_id)
        if status.phase == ContainerPhase.STOPPED:
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

    def __init__(self, cache_dir: Path, capacity_type: CapacityType | None = None) -> None:
        self._cache_dir = cache_dir
        # Drives whether per-container CPU is a hard cap (`--cpus`) or a soft
        # weight (`--cpu-shares`). On-demand workers use soft weights so small
        # entrypoint/coordinator containers can burst onto otherwise-idle host
        # CPU; preemptible/reserved keep the hard cap for predictability.
        self.capacity_type = capacity_type
        self._handles: list[DockerContainerHandle] = []
        self._created_containers: set[str] = set()
        # Serializes `docker pull` per image tag so that concurrent task threads
        # don't each trigger docker-credential-gcloud against the metadata server,
        # which causes sporadic "no active account" errors under load.
        self._pull_lock = threading.Lock()
        self._pulled_images: set[str] = set()

    def ensure_image(self, image: str) -> None:
        """Pull *image* if it isn't already present locally.

        Only one `docker pull` runs at a time (via ``_pull_lock``).  This
        prevents a thundering-herd of ``docker-credential-gcloud`` processes
        when many task threads call ``docker create`` concurrently — each
        invocation would otherwise shell out to the GCE metadata server for an
        OAuth token, overwhelming it and causing sporadic auth failures.
        """
        if image in self._pulled_images:
            return

        with self._pull_lock:
            # Double-check after acquiring lock
            if image in self._pulled_images:
                return

            # Fast path: image already on disk
            inspect = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                check=False,
            )
            if inspect.returncode == 0:
                self._pulled_images.add(image)
                return

            logger.info("Pulling image %s", image)
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr
                if _is_docker_infra_error(stderr):
                    raise ContainerInfraError(f"Failed to pull image {image} (infra): {stderr}")
                raise RuntimeError(f"Failed to pull image {image}: {stderr}")

            logger.info("Image %s pulled successfully", image)
            self._pulled_images.add(image)

    def resolve_mounts(self, mounts: list[MountSpec], workdir_host_path: Path | None = None) -> list[ResolvedMount]:
        """Convert semantic MountSpecs to ResolvedMount instances.

        Creates host directories as needed. WORKDIR uses the explicit host path
        (created by task_attempt). CACHE gets shared dirs under cache_dir.
        TMPFS uses Docker --tmpfs for per-container isolation (no host dir).
        """
        result: list[ResolvedMount] = []
        for mount in mounts:
            mode = "ro" if mount.read_only else "rw"
            if mount.kind == MountKind.WORKDIR:
                if workdir_host_path is None:
                    raise RuntimeError("WORKDIR mount requires workdir_host_path")
                result.append(ResolvedMount(str(workdir_host_path), mount.container_path, mode, mount.kind))
            elif mount.kind == MountKind.TMPFS:
                # TMPFS mounts use Docker --tmpfs (per-container isolation); no host dir needed
                result.append(ResolvedMount("", mount.container_path, mode, mount.kind))
            elif mount.kind == MountKind.CACHE:
                host_dir = self._cache_dir / mount.container_path.strip("/").replace("/", "-")
                host_dir.mkdir(parents=True, exist_ok=True)
                result.append(ResolvedMount(str(host_dir), mount.container_path, mode, mount.kind))
        return result

    def create_container(self, config: ContainerConfig) -> DockerContainerHandle:
        """Create a container handle from config.

        The handle is not started - call handle.build() then handle.run()
        to execute the container.
        """
        resolved = self.resolve_mounts(config.mounts, workdir_host_path=config.workdir_host_path)
        handle = DockerContainerHandle(config=config, runtime=self, _resolved_mounts=resolved)
        self._handles.append(handle)
        return handle

    def prepare_workdir(self, workdir: Path, disk_bytes: int) -> None:
        """No-op: workdirs live on cache_dir (/dev/shm/iris) which is already tmpfs."""

    def stage_bundle(
        self,
        *,
        bundle_id: str,
        workdir: Path,
        workdir_files: dict[str, bytes],
        bundle_store: BundleStore,
    ) -> None:
        """Stage bundle and workdir files on worker-local filesystem."""
        if bundle_id:
            bundle_store.extract_bundle_to(bundle_id, workdir)
        write_workdir_files(workdir, workdir_files)

    def track_container(self, container_id: str) -> None:
        """Track a container ID for cleanup."""
        self._created_containers.add(container_id)

    def untrack_container(self, container_id: str) -> None:
        """Untrack a container ID."""
        self._created_containers.discard(container_id)

    def list_iris_containers(self, all_states: bool = True) -> list[str]:
        """List all containers with iris.managed=true label."""
        cmd = ["docker", "ps", "-q", "--filter", "label=iris.managed=true"]
        if all_states:
            cmd.append("-a")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return []
        return [cid for cid in result.stdout.strip().split("\n") if cid]

    def discover_containers(self) -> list[DiscoveredContainer]:
        """Discover iris-managed containers from a previous worker process.

        Inspects all iris-managed containers and extracts metadata from labels
        and state. Used during worker restart to find running containers that
        can be adopted instead of killed.
        """
        container_ids = self.list_iris_containers(all_states=True)
        if not container_ids:
            return []

        # Batch inspect all containers in one call
        result = subprocess.run(
            ["docker", "inspect", *container_ids],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("docker inspect failed during discovery: %s", result.stderr)
            return []

        discovered: list[DiscoveredContainer] = []
        for info in json.loads(result.stdout):
            labels = info.get("Config", {}).get("Labels", {})
            state = info.get("State", {})

            task_id = labels.get("iris.task_id", "")
            attempt_id_str = labels.get("iris.attempt_id")
            if not task_id or attempt_id_str is None:
                # Missing required labels — cannot adopt
                continue

            # Find the /app mount's host path
            workdir_host_path = ""
            for mount in info.get("Mounts", []):
                if mount.get("Destination") == "/app":
                    workdir_host_path = mount.get("Source", "")
                    break

            discovered.append(
                DiscoveredContainer(
                    container_id=info["Id"],
                    task_id=task_id,
                    attempt_id=int(attempt_id_str),
                    attempt_uid=labels.get("iris.attempt_uid", ""),
                    job_id=labels.get("iris.job_id", ""),
                    worker_id=labels.get("iris.worker_id", ""),
                    phase=ExecutionStage(labels.get("iris.phase", "run")),
                    running=state.get("Running", False),
                    exit_code=state.get("ExitCode") if not state.get("Running", False) else None,
                    started_at=state.get("StartedAt", ""),
                    workdir_host_path=workdir_host_path,
                    ports=json.loads(labels.get("iris.ports", "{}")),
                )
            )

        return discovered

    def adopt_container(self, container_id: str) -> DockerContainerHandle:
        """Wrap an existing container for adoption after worker restart."""
        return DockerContainerHandle.from_existing(container_id, self)

    def remove_containers(self, container_ids: list[str]) -> int:
        """Force remove specific containers by ID. Returns count removed."""
        if not container_ids:
            return 0
        subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            capture_output=True,
            check=False,
        )
        return len(container_ids)

    def remove_all_iris_containers(self) -> int:
        """Force remove all iris-managed containers. Returns count attempted."""
        return self.remove_containers(self.list_iris_containers(all_states=True))

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
