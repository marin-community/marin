# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared profiling command construction for CPU (py-spy), memory (memray), and threads.

Both the Docker and process runtimes build their profiler commands through
this module, eliminating duplicated format maps and fragile index-based
command construction.

The module also provides `profile_local_process` for profiling the current
interpreter process (used by the controller and worker for /system/process).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from iris.cluster.types import TaskAttempt
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# Target sentinel for profiling the local worker/controller process itself.
SYSTEM_PROCESS_TARGET = "/system/process"

# Shell expression that resolves to the oldest python PID inside a task
# container, falling back to PID 1 if none is found. Task entrypoints are
# typically `uv run python ...`, so PID 1 is the `uv` Rust binary and py-spy
# cannot fingerprint it as CPython. The user's actual interpreter is a child
# of uv; `pgrep -o python` selects the oldest matching process (the main
# interpreter), and py-spy's --subprocesses then walks any further children.
# The expression must be evaluated inside a shell on the target container.
PYTHON_PID_EXPR = "$(pgrep -o python || echo 1)"

CPU_FORMAT_MAP: dict[int, tuple[str, str]] = {
    job_pb2.CpuProfile.FLAMEGRAPH: ("flamegraph", "svg"),
    job_pb2.CpuProfile.SPEEDSCOPE: ("speedscope", "json"),
    job_pb2.CpuProfile.RAW: ("raw", "txt"),
}

MEMORY_FORMAT_MAP: dict[int, tuple[str, str]] = {
    job_pb2.MemoryProfile.FLAMEGRAPH: ("flamegraph", "html"),
    job_pb2.MemoryProfile.TABLE: ("table", "txt"),
    job_pb2.MemoryProfile.STATS: ("stats", "json"),
    job_pb2.MemoryProfile.RAW: ("raw", "bin"),
}


@dataclass(frozen=True)
class CpuProfileSpec:
    py_spy_format: str
    ext: str
    rate_hz: int
    duration_seconds: int
    pid: str
    native: bool = True


@dataclass(frozen=True)
class MemoryProfileSpec:
    reporter: str  # "flamegraph", "table", "stats"
    ext: str
    duration_seconds: int
    pid: str
    leaks: bool

    @property
    def is_raw(self) -> bool:
        """Raw mode returns the .bin trace directly, skipping transform."""
        return self.reporter == "raw"

    @property
    def output_is_file(self) -> bool:
        """Flamegraph and stats write to a file; table writes to stdout."""
        return self.reporter in ("flamegraph", "stats")


def resolve_cpu_spec(cpu_config: job_pb2.CpuProfile, duration_seconds: int, pid: str) -> CpuProfileSpec:
    py_spy_format, ext = CPU_FORMAT_MAP.get(cpu_config.format, ("flamegraph", "svg"))
    rate_hz = cpu_config.rate_hz if cpu_config.rate_hz > 0 else 20
    native = cpu_config.native if cpu_config.HasField("native") else True
    return CpuProfileSpec(
        py_spy_format=py_spy_format,
        ext=ext,
        rate_hz=rate_hz,
        duration_seconds=duration_seconds,
        pid=pid,
        native=native,
    )


def resolve_memory_spec(memory_config: job_pb2.MemoryProfile, duration_seconds: int, pid: str) -> MemoryProfileSpec:
    reporter, ext = MEMORY_FORMAT_MAP.get(memory_config.format, ("flamegraph", "html"))
    return MemoryProfileSpec(
        reporter=reporter,
        ext=ext,
        duration_seconds=duration_seconds,
        pid=pid,
        leaks=memory_config.leaks,
    )


def build_pyspy_cmd(spec: CpuProfileSpec, py_spy_bin: str, output_path: str) -> list[str]:
    return [
        py_spy_bin,
        "record",
        "--pid",
        spec.pid,
        "--duration",
        str(spec.duration_seconds),
        "--rate",
        str(spec.rate_hz),
        "--format",
        spec.py_spy_format,
        "--output",
        output_path,
        "--subprocesses",
        *(["--native"] if spec.native else []),
    ]


def build_memray_attach_cmd(spec: MemoryProfileSpec, memray_bin: str, trace_path: str) -> list[str]:
    cmd = [memray_bin, "attach", "--native", spec.pid, "--duration", str(spec.duration_seconds), "--output", trace_path]
    if spec.leaks:
        cmd.append("--aggregate")
    return cmd


def build_memray_transform_cmd(spec: MemoryProfileSpec, memray_bin: str, trace_path: str, output_path: str) -> list[str]:
    """Build the memray transform command.

    For flamegraph/stats, writes to output_path. For table, output goes to stdout.
    """
    if spec.reporter == "flamegraph":
        cmd = [memray_bin, "flamegraph"]
        if spec.leaks:
            cmd.append("--leaks")
        cmd.extend(["--force", "--output", output_path, trace_path])
        return cmd
    elif spec.reporter == "table":
        return [memray_bin, "table", trace_path]
    elif spec.reporter == "stats":
        return [memray_bin, "stats", "--json", "--force", "-o", output_path, trace_path]
    else:
        raise RuntimeError(f"Unknown memray reporter: {spec.reporter}")


def build_pyspy_dump_cmd(pid: str, py_spy_bin: str = "py-spy", *, include_locals: bool = False) -> list[str]:
    """Build a py-spy dump command for thread-level stack traces."""
    cmd = [py_spy_bin, "dump", "--pid", pid, "--subprocesses"]
    if include_locals:
        cmd.append("--locals")
    return cmd


def profile_local_process(duration_seconds: int, profile_type: job_pb2.ProfileType) -> bytes:
    """Profile the current interpreter process using py-spy or memray.

    Used by the controller and worker to handle /system/process targets.
    Raises RuntimeError immediately if the required tool is not installed.
    """
    pid = str(os.getpid())

    if profile_type.HasField("threads"):
        _check_tool("py-spy")
        return run_pyspy_dump(pid, include_locals=profile_type.threads.locals)
    elif profile_type.HasField("cpu"):
        _check_tool("py-spy")
        return _run_pyspy_record(pid, duration_seconds, profile_type.cpu)
    elif profile_type.HasField("memory"):
        _check_tool("memray")
        return _run_memray_profile(pid, duration_seconds, profile_type.memory)
    else:
        raise RuntimeError("ProfileType must specify cpu, memory, or threads profiler")


def run_pyspy_dump(pid: str, py_spy_bin: str = "py-spy", *, include_locals: bool = False) -> bytes:
    """Run py-spy dump to collect thread stacks from a process."""
    cmd = build_pyspy_dump_cmd(pid, py_spy_bin, include_locals=include_locals)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"py-spy dump failed: {result.stderr}")
    return result.stdout.encode("utf-8")


def _run_pyspy_record(pid: str, duration_seconds: int, cpu_config: job_pb2.CpuProfile) -> bytes:
    """Run py-spy record against a local process and return the output."""
    spec = resolve_cpu_spec(cpu_config, duration_seconds, pid=pid)
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{spec.ext}", delete=False) as f:
            output_path = f.name

        cmd = build_pyspy_cmd(spec, py_spy_bin="py-spy", output_path=output_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_seconds + 30)
        if result.returncode != 0:
            raise RuntimeError(f"py-spy record failed: {result.stderr}")
        return Path(output_path).read_bytes()
    finally:
        if output_path is not None:
            Path(output_path).unlink(missing_ok=True)


def _run_memray_profile(pid: str, duration_seconds: int, memory_config: job_pb2.MemoryProfile) -> bytes:
    """Profile memory of the current process using memray's in-process Tracker.

    Uses the programmatic Tracker API instead of ``memray attach``, avoiding
    ptrace/SYS_PTRACE requirements that fail when profiling the controller or
    worker's own process from within a container.
    """
    import memray

    spec = resolve_memory_spec(memory_config, duration_seconds, pid=pid)
    file_format = memray.FileFormat.AGGREGATED_ALLOCATIONS if spec.leaks else memray.FileFormat.ALL_ALLOCATIONS

    trace_path = None
    output_path = None
    try:
        # Tracker refuses to overwrite an existing file, so get a unique name
        # then remove the placeholder before passing to Tracker.
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            trace_path = f.name
        os.unlink(trace_path)

        # Track allocations in-process for the requested duration.
        with memray.Tracker(trace_path, native_traces=True, file_format=file_format):
            time.sleep(duration_seconds)

        # Raw mode: return the .bin trace directly, no transform needed.
        if spec.is_raw:
            return Path(trace_path).read_bytes()

        if spec.output_is_file:
            with tempfile.NamedTemporaryFile(suffix=f".{spec.ext}", delete=False) as f:
                output_path = f.name

        transform_cmd = build_memray_transform_cmd(
            spec, memray_bin="memray", trace_path=trace_path, output_path=output_path or ""
        )
        result = subprocess.run(transform_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"memray {spec.reporter} failed: {result.stderr}")

        if spec.output_is_file:
            return Path(output_path).read_bytes()
        else:
            return result.stdout.encode("utf-8")
    finally:
        if trace_path is not None:
            Path(trace_path).unlink(missing_ok=True)
        if output_path is not None:
            Path(output_path).unlink(missing_ok=True)


def is_system_target(target: str) -> bool:
    """Return True if the target refers to the local process rather than a task."""
    return target == SYSTEM_PROCESS_TARGET


def parse_profile_target(target: str) -> TaskAttempt:
    """Parse a task target string into a TaskAttempt.

    Examples:
        >>> parse_profile_target("/alice/job/0")
        TaskAttempt('/alice/job/0')
        >>> parse_profile_target("/alice/job/0:3")
        TaskAttempt('/alice/job/0:3')
    """
    return TaskAttempt.from_wire(target)


def _check_tool(name: str) -> None:
    """Raise RuntimeError if a profiling tool is not on PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"'{name}' is not installed. /system/process profiling requires "
            f"py-spy and memray to be available in the controller/worker environment."
        )
