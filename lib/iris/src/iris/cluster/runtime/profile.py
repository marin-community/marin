# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared profiling command construction for CPU (py-spy), memory (memray), and threads.

Both the Docker and process runtimes build their profiler commands through
this module, eliminating duplicated format maps and fragile index-based
command construction.

The module also provides `profile_local_process` for profiling the current
interpreter process (used by the controller and worker for /system/process).
"""

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from iris.cluster.stats.tables import IrisProfile, ProfileFormat, ProfileTrigger, ProfileType
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# Target sentinel for profiling the local worker/controller process itself.
SYSTEM_PROCESS_TARGET = "/system/process"


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


def build_pyspy_cmd(spec: CpuProfileSpec, py_spy_bin: str, output_path: str, *, subprocesses: bool = True) -> list[str]:
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
        *(["--subprocesses"] if subprocesses else []),
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


def build_pyspy_dump_cmd(
    pid: str, py_spy_bin: str = "py-spy", *, include_locals: bool = False, subprocesses: bool = True
) -> list[str]:
    """Build a py-spy dump command for thread-level stack traces."""
    cmd = [py_spy_bin, "dump", "--pid", pid]
    if subprocesses:
        cmd.append("--subprocesses")
    if include_locals:
        cmd.append("--locals")
    return cmd


# Workaround for https://github.com/benfred/py-spy/issues/846: py-spy dump --subprocesses
# exits non-zero when it walks into a non-Python child (e.g. wandb-core, a Go binary), even
# though the parent dump already went to stdout. Drop this once py-spy fixes the upstream bug.
_PYSPY_NON_PYTHON_CHILD_ERROR = "Failed to find python version from target process"

# Headroom a healthy profiler needs beyond the sample window to attach and write
# its output. These set the in-environment watchdog deadline; the dispatcher's
# own client-side timeout is a few seconds longer (see PROFILER_WATCHDOG_GRACE_SECONDS).
CPU_WRITE_HEADROOM_SECONDS = 30
MEMRAY_ATTACH_HEADROOM_SECONDS = 30
THREAD_DUMP_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# ptrace profiler safety: watchdog + group-stop recovery
# ---------------------------------------------------------------------------
# py-spy and ``memray attach`` attach to the target via ptrace, which pauses it
# while sampling. Two failure modes can freeze the target indefinitely when the
# profiler is launched into an isolated PID namespace over a client connection
# (``docker exec`` / ``kubectl exec``):
#
#   1. Orphaned profiler. A client-side ``subprocess.run(timeout=...)`` only
#      kills the local exec *client*; the in-namespace profiler keeps running
#      and keeps the target ptrace-stopped (observed: zephyr shards stalled for
#      hours). Wrapping the profiler in GNU ``timeout --signal=KILL`` makes the
#      container/pod reap it regardless of the client.
#
#   2. Lingering group-stop. py-spy attaches via ``PTRACE_ATTACH``, which
#      delivers SIGSTOP. Per ptrace(2), a SIGKILL'd tracer only auto-resumes a
#      *ptrace*-stopped tracee, not a *group*-stopped one, so a profiler that
#      dies or exits before detaching cleanly can leave the target in
#      job-control stop (``T``), needing SIGCONT (benfred/py-spy#390, still seen
#      on recent releases).
#
# Both are specific to the exec-into-namespace model and are handled inside the
# namespaced dispatchers' ``exec_profiler``. A direct ``subprocess.run`` profiler
# (process runtime, controller self-profile) is reaped by Python's own timeout
# and shares the host PID namespace, so it needs no ``timeout`` wrapper and
# recovers (if at all) by SIGCONT-ing the profiled child's process group.
PROFILER_WATCHDOG_GRACE_SECONDS = 5


def wrap_with_kill_watchdog(cmd: list[str], sample_timeout: int) -> list[str]:
    """Prefix a profiler command with ``timeout --signal=KILL`` so a hung profiler self-reaps."""
    return ["timeout", "--signal=KILL", str(sample_timeout), *cmd]


def sigcont_sweep_argv() -> list[str]:
    """Command that SIGCONTs every process in the (container/pod) PID namespace.

    Clears a group-stop a ptrace profiler may have left. Sweeps ``/proc`` so it
    needs no procps, reaches ``--subprocesses`` children, and signals PID 1
    (which the ``kill -1`` broadcast deliberately skips). SIGCONT is a no-op on
    running processes. Only safe where the PID namespace is isolated.
    """
    return ["sh", "-c", 'for p in /proc/[0-9]*; do kill -CONT "${p##*/}" 2>/dev/null; done; true']


@dataclass(frozen=True)
class ExecResult:
    """Normalized result of running a command in a target environment."""

    returncode: int
    stdout: bytes
    stderr: str


class ProfileDispatch(Protocol):
    """Where/how a profiling backend runs commands and moves files.

    The ``capture_*`` functions own *what* to run (which py-spy/memray command,
    when to read output, how to interpret exit codes); a dispatch implementation
    owns *where* it runs — ``docker exec``, ``kubectl exec``, or a host
    subprocess. ``exec_profiler`` runs a ptrace-attaching profiler and is
    responsible for reaping a hung profiler and clearing any group-stop it leaves
    (see the module notes above); ``exec`` runs non-attaching helpers (e.g.
    ``memray`` transform) that never touch the live target.
    """

    pyspy_bin: str
    memray_bin: str

    def scratch(self, *suffixes: str) -> AbstractContextManager[tuple[str, ...]]:
        """Yield one temp path per suffix, removing them all on context exit."""
        ...

    def exec_profiler(self, cmd: list[str], *, sample_timeout: int) -> ExecResult:
        """Run a ptrace-attaching profiler, reaping it and clearing any group-stop."""
        ...

    def exec(self, cmd: list[str], *, timeout: int) -> ExecResult:
        """Run a non-attaching helper command."""
        ...

    def read_file(self, path: str) -> bytes:
        """Read a file produced by a profiler in the target environment."""
        ...


def capture_cpu(
    dispatch: ProfileDispatch,
    cpu_config: job_pb2.CpuProfile,
    duration_seconds: int,
    *,
    pid: str,
    subprocesses: bool = True,
) -> bytes:
    """Record a CPU profile with py-spy and return the encoded output bytes."""
    spec = resolve_cpu_spec(cpu_config, duration_seconds, pid=pid)
    with dispatch.scratch(spec.ext) as (output_path,):
        cmd = build_pyspy_cmd(spec, dispatch.pyspy_bin, output_path, subprocesses=subprocesses)
        result = dispatch.exec_profiler(cmd, sample_timeout=duration_seconds + CPU_WRITE_HEADROOM_SECONDS)
        if result.returncode != 0:
            raise RuntimeError(f"py-spy record failed (exit {result.returncode}): {result.stderr}")
        return dispatch.read_file(output_path)


def capture_threads(
    dispatch: ProfileDispatch, *, pid: str, include_locals: bool = False, subprocesses: bool = True
) -> bytes:
    """Collect thread stacks with py-spy dump and return the raw stdout bytes."""
    cmd = build_pyspy_dump_cmd(pid, dispatch.pyspy_bin, include_locals=include_locals, subprocesses=subprocesses)
    result = dispatch.exec_profiler(cmd, sample_timeout=THREAD_DUMP_TIMEOUT_SECONDS)
    if result.returncode != 0:
        partial_ok = subprocesses and bool(result.stdout.strip()) and _PYSPY_NON_PYTHON_CHILD_ERROR in result.stderr
        if not partial_ok:
            raise RuntimeError(f"py-spy dump failed (exit {result.returncode}): {result.stderr}")
    return result.stdout


def capture_memory_attach(
    dispatch: ProfileDispatch, memory_config: job_pb2.MemoryProfile, duration_seconds: int, *, pid: str
) -> bytes:
    """Profile memory by attaching memray to a running process, then transforming the trace."""
    spec = resolve_memory_spec(memory_config, duration_seconds, pid=pid)
    with dispatch.scratch("bin", spec.ext) as (trace_path, output_path):
        attach_cmd = build_memray_attach_cmd(spec, dispatch.memray_bin, trace_path)
        result = dispatch.exec_profiler(attach_cmd, sample_timeout=duration_seconds + MEMRAY_ATTACH_HEADROOM_SECONDS)
        if result.returncode != 0:
            raise RuntimeError(f"memray attach failed (exit {result.returncode}): {result.stderr}")

        if spec.is_raw:
            return dispatch.read_file(trace_path)

        transform_cmd = build_memray_transform_cmd(
            spec, dispatch.memray_bin, trace_path, output_path if spec.output_is_file else ""
        )
        result = dispatch.exec(transform_cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"memray {spec.reporter} failed (exit {result.returncode}): {result.stderr}")

        return dispatch.read_file(output_path) if spec.output_is_file else result.stdout


@dataclass
class LocalProfileDispatch:
    """Run profilers as host subprocesses, sharing the host PID namespace.

    Python's own subprocess timeout reaps a hung profiler (no orphan), so no
    ``timeout`` wrapper is used. ``resume_pid`` (when set) is SIGCONT'd by
    process group after each attach to clear a py-spy group-stop; self-profiling
    leaves it ``None`` because a group-stopped process cannot signal itself.
    """

    pyspy_bin: str = "py-spy"
    memray_bin: str = "memray"
    resume_pid: int | None = None

    @contextmanager
    def scratch(self, *suffixes: str) -> Iterator[tuple[str, ...]]:
        paths = []
        for suffix in suffixes:
            with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as f:
                paths.append(f.name)
        try:
            yield tuple(paths)
        finally:
            for path in paths:
                Path(path).unlink(missing_ok=True)

    def exec_profiler(self, cmd: list[str], *, sample_timeout: int) -> ExecResult:
        try:
            return self.exec(cmd, timeout=sample_timeout)
        finally:
            self._resume()

    def exec(self, cmd: list[str], *, timeout: int) -> ExecResult:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return ExecResult(result.returncode, (result.stdout or "").encode("utf-8"), result.stderr or "")

    def read_file(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def _resume(self) -> None:
        if self.resume_pid is None or sys.platform != "linux":
            return
        try:
            os.killpg(os.getpgid(self.resume_pid), signal.SIGCONT)
        except (ProcessLookupError, PermissionError):
            pass


def profile_local_process(duration_seconds: int, profile_type: job_pb2.ProfileType) -> bytes:
    """Profile the current interpreter process using py-spy or memray.

    Used by the controller and worker to handle /system/process targets.
    Raises RuntimeError immediately if the required tool is not installed.
    """
    pid = str(os.getpid())
    # Self-profiling shares this process's PID namespace and cannot SIGCONT
    # itself, so resume_pid stays None and --subprocesses is off.
    dispatch = LocalProfileDispatch()

    if profile_type.HasField("threads"):
        _check_tool("py-spy")
        return capture_threads(dispatch, pid=pid, include_locals=profile_type.threads.locals, subprocesses=False)
    elif profile_type.HasField("cpu"):
        _check_tool("py-spy")
        return capture_cpu(dispatch, profile_type.cpu, duration_seconds, pid=pid, subprocesses=False)
    elif profile_type.HasField("memory"):
        _check_tool("memray")
        return _run_memray_profile(pid, duration_seconds, profile_type.memory)
    else:
        raise RuntimeError("ProfileType must specify cpu, memory, or threads profiler")


def _run_memray_profile(pid: str, duration_seconds: int, memory_config: job_pb2.MemoryProfile) -> bytes:
    """Profile memory of the current process using memray's in-process Tracker.

    Uses the programmatic Tracker API instead of ``memray attach``, avoiding
    ptrace/SYS_PTRACE requirements that fail when profiling the controller or
    worker's own process from within a container.
    """
    import memray  # noqa: PLC0415  # optional dep: memray

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


def _check_tool(name: str) -> None:
    """Raise RuntimeError if a profiling tool is not on PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"'{name}' is not installed. /system/process profiling requires "
            f"py-spy and memray to be available in the controller/worker environment."
        )


_MEMORY_REPORTER_TO_FORMAT: dict[str, ProfileFormat] = {
    "flamegraph": ProfileFormat.HTML,
    "table": ProfileFormat.TABLE,
    "stats": ProfileFormat.STATS,
    "raw": ProfileFormat.RAW,
}


def build_profile_row(
    *,
    source: str,
    attempt_id: int | None,
    vm_id: str,
    duration_seconds: int,
    profile_type: job_pb2.ProfileType,
    profile_data: bytes,
    trigger: ProfileTrigger = ProfileTrigger.ON_DEMAND,
) -> IrisProfile:
    """Construct one ``IrisProfile`` row from a completed capture.

    Single source of truth for the worker, ``K8sTaskProvider``, and the
    controller's ``/system/controller`` self-capture path. ``type``,
    ``format``, and the type-specific metadata fields are derived from the
    proto oneof — callers only supply identity (``source`` / ``attempt_id``
    / ``vm_id``), the captured bytes, and the trigger.
    """
    captured_at = datetime.now(UTC).replace(tzinfo=None)
    which = profile_type.WhichOneof("profiler")
    if which == "cpu":
        cpu_spec = resolve_cpu_spec(profile_type.cpu, duration_seconds, pid="")
        return IrisProfile(
            source=source,
            attempt_id=attempt_id,
            vm_id=vm_id,
            captured_at=captured_at,
            duration_seconds=duration_seconds,
            type=ProfileType.CPU.value,
            format=cpu_spec.py_spy_format,
            trigger=trigger.value,
            rate_hz=cpu_spec.rate_hz,
            native=cpu_spec.native,
            profile_data=profile_data,
        )
    if which == "memory":
        memory_spec = resolve_memory_spec(profile_type.memory, duration_seconds, pid="")
        fmt = _MEMORY_REPORTER_TO_FORMAT.get(memory_spec.reporter, ProfileFormat.HTML)
        return IrisProfile(
            source=source,
            attempt_id=attempt_id,
            vm_id=vm_id,
            captured_at=captured_at,
            duration_seconds=duration_seconds,
            type=ProfileType.MEMORY.value,
            format=fmt.value,
            trigger=trigger.value,
            leaks=memory_spec.leaks,
            profile_data=profile_data,
        )
    if which == "threads":
        return IrisProfile(
            source=source,
            attempt_id=attempt_id,
            vm_id=vm_id,
            captured_at=captured_at,
            duration_seconds=duration_seconds,
            type=ProfileType.THREAD.value,
            format=ProfileFormat.RAW.value,
            trigger=trigger.value,
            locals_dump=bool(profile_type.threads.locals),
            profile_data=profile_data,
        )
    raise ValueError(f"ProfileType has no profiler set: {profile_type!r}")
