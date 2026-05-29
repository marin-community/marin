# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for profile command construction helpers.

Focuses on format-to-flag mapping, default handling, and CLI structure —
not on pass-through of constructor arguments.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field

import pytest
from iris.cluster.runtime.profile import (
    ExecResult,
    _run_memray_profile,
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    capture_cpu,
    capture_memory_attach,
    capture_threads,
    resolve_cpu_spec,
    resolve_memory_spec,
    sigcont_sweep_argv,
    wrap_with_kill_watchdog,
)
from iris.rpc import job_pb2

memray = pytest.importorskip("memray")

# ---------------------------------------------------------------------------
# resolve_cpu_spec: enum → (py_spy_format, ext) mapping and defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proto_format, expected_format, expected_ext",
    [
        (job_pb2.CpuProfile.FLAMEGRAPH, "flamegraph", "svg"),
        (job_pb2.CpuProfile.SPEEDSCOPE, "speedscope", "json"),
        (job_pb2.CpuProfile.RAW, "raw", "txt"),
    ],
)
def test_resolve_cpu_spec_maps_format_to_pyspy_format_and_extension(proto_format, expected_format, expected_ext):
    cfg = job_pb2.CpuProfile(format=proto_format, rate_hz=100)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    assert spec.py_spy_format == expected_format
    assert spec.ext == expected_ext


def test_resolve_cpu_spec_defaults_rate_hz_when_zero():
    cfg = job_pb2.CpuProfile(format=job_pb2.CpuProfile.FLAMEGRAPH, rate_hz=0)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    assert spec.rate_hz == 20


def test_resolve_cpu_spec_preserves_nonzero_rate_hz():
    cfg = job_pb2.CpuProfile(format=job_pb2.CpuProfile.FLAMEGRAPH, rate_hz=250)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    assert spec.rate_hz == 250


# ---------------------------------------------------------------------------
# resolve_memory_spec: enum → (reporter, ext) mapping and output_is_file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proto_format, expected_reporter, expected_ext, expected_is_file",
    [
        (job_pb2.MemoryProfile.FLAMEGRAPH, "flamegraph", "html", True),
        (job_pb2.MemoryProfile.TABLE, "table", "txt", False),
        (job_pb2.MemoryProfile.STATS, "stats", "json", True),
        (job_pb2.MemoryProfile.RAW, "raw", "bin", False),
    ],
)
def test_resolve_memory_spec_maps_format(proto_format, expected_reporter, expected_ext, expected_is_file):
    cfg = job_pb2.MemoryProfile(format=proto_format)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    assert spec.reporter == expected_reporter
    assert spec.ext == expected_ext
    assert spec.output_is_file is expected_is_file


# ---------------------------------------------------------------------------
# build_pyspy_cmd: verify CLI flag structure
# ---------------------------------------------------------------------------


def test_build_pyspy_cmd_includes_subprocesses_flag_by_default():
    cfg = job_pb2.CpuProfile(format=job_pb2.CpuProfile.FLAMEGRAPH, rate_hz=100)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_pyspy_cmd(spec, py_spy_bin="py-spy", output_path="/tmp/out.svg")
    assert "--subprocesses" in cmd


# ---------------------------------------------------------------------------
# build_memray_attach_cmd: --aggregate flag depends on leaks
# ---------------------------------------------------------------------------


def test_memray_attach_includes_aggregate_when_leaks_enabled():
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH, leaks=True)
    spec = resolve_memory_spec(cfg, duration_seconds=10, pid="5")
    cmd = build_memray_attach_cmd(spec, memray_bin="memray", trace_path="/tmp/trace.bin")
    assert "--aggregate" in cmd
    assert "--native" in cmd


def test_memray_attach_excludes_aggregate_when_leaks_disabled():
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.TABLE, leaks=False)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_attach_cmd(spec, memray_bin="memray", trace_path="/tmp/trace.bin")
    assert "--aggregate" not in cmd
    assert "--native" in cmd


# ---------------------------------------------------------------------------
# build_memray_transform_cmd: reporter determines subcommand and output mode
# ---------------------------------------------------------------------------


def test_memray_transform_flamegraph_writes_to_file_with_leaks():
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH, leaks=True)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="/tmp/o.html")

    assert "flamegraph" in cmd
    assert "--leaks" in cmd
    assert "--output" in cmd
    assert "/tmp/o.html" in cmd


def test_memray_transform_table_does_not_write_to_file():
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.TABLE)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="")

    assert "--output" not in cmd


def test_memray_transform_stats_includes_json_flag_and_output():
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.STATS)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="/tmp/o.json")

    assert "stats" in cmd
    assert "--json" in cmd
    assert "-o" in cmd
    assert "/tmp/o.json" in cmd


# ---------------------------------------------------------------------------
# _run_memray_profile: in-process Tracker produces valid output
# ---------------------------------------------------------------------------


def _allocate_during(duration_seconds: int) -> list:
    """Force allocations so memray captures something during short profiles."""

    results: list = []

    def _alloc():
        for _ in range(duration_seconds * 100):
            results.append(bytearray(1024))

            time.sleep(duration_seconds / 100)

    t = threading.Thread(target=_alloc, daemon=True)
    t.start()
    return results


def test_resolve_memory_spec_raw_is_raw():
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.RAW)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    assert spec.is_raw is True


def test_resolve_memory_spec_flamegraph_is_not_raw():
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    assert spec.is_raw is False


@pytest.mark.parametrize(
    "proto_format",
    [
        job_pb2.MemoryProfile.FLAMEGRAPH,
        job_pb2.MemoryProfile.TABLE,
        job_pb2.MemoryProfile.STATS,
        job_pb2.MemoryProfile.RAW,
    ],
)
def test_run_memray_profile_returns_nonempty_output(proto_format):
    """In-process memray Tracker produces non-empty output for flamegraph/table/stats."""
    _allocate_during(1)
    cfg = job_pb2.MemoryProfile(format=proto_format, leaks=False)
    pid = str(os.getpid())
    result = _run_memray_profile(pid, duration_seconds=1, memory_config=cfg)
    assert len(result) > 0


def test_run_memray_profile_stats_returns_valid_json():
    """Stats reporter returns parseable JSON, not a file-path string."""
    _allocate_during(1)
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.STATS, leaks=False)
    pid = str(os.getpid())
    result = _run_memray_profile(pid, duration_seconds=1, memory_config=cfg)
    data = json.loads(result)
    assert "total_num_allocations" in data


# ---------------------------------------------------------------------------
# Watchdog / recovery command construction
# ---------------------------------------------------------------------------


def test_wrap_with_kill_watchdog_prefixes_timeout_sigkill():
    wrapped = wrap_with_kill_watchdog(["py-spy", "record"], sample_timeout=42)
    assert wrapped == ["timeout", "--signal=KILL", "42", "py-spy", "record"]


def test_sigcont_sweep_continues_every_pid_via_proc():
    argv = sigcont_sweep_argv()
    assert argv[0] == "sh"
    # Sweeps /proc (no procps) and SIGCONTs each pid, including PID 1.
    assert "/proc/[0-9]*" in argv[-1]
    assert "kill -CONT" in argv[-1]


# ---------------------------------------------------------------------------
# Shared capture_* orchestration, exercised through a fake dispatch
# ---------------------------------------------------------------------------


@dataclass
class FakeDispatch:
    """In-memory ProfileDispatch: records commands and serves canned files."""

    pyspy_bin: str = "py-spy"
    memray_bin: str = "memray"
    files: dict[str, bytes] = field(default_factory=dict)
    profiler_result: ExecResult = field(default_factory=lambda: ExecResult(0, b"", ""))
    transform_result: ExecResult = field(default_factory=lambda: ExecResult(0, b"", ""))
    profiler_cmds: list[list[str]] = field(default_factory=list)
    transform_cmds: list[list[str]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    _counter: int = 0

    def tmp_path(self, suffix: str) -> str:
        self._counter += 1
        return f"/tmp/fake-{self._counter}.{suffix}"

    def exec_profiler(self, cmd, *, sample_timeout):
        self.profiler_cmds.append(cmd)
        return self.profiler_result

    def exec(self, cmd, *, timeout):
        self.transform_cmds.append(cmd)
        return self.transform_result

    def read_file(self, path):
        return self.files[path]

    def rm_files(self, paths):
        self.removed.extend(paths)


def test_capture_cpu_records_reads_and_cleans_up():
    dispatch = FakeDispatch()
    # py-spy writes to the dispatch-chosen output path; serve its bytes back.
    dispatch.files["/tmp/fake-1.json"] = b"speedscope-bytes"
    cfg = job_pb2.CpuProfile(format=job_pb2.CpuProfile.SPEEDSCOPE)

    data = capture_cpu(dispatch, cfg, duration_seconds=5, pid="1")

    assert data == b"speedscope-bytes"
    assert dispatch.profiler_cmds[0][:2] == ["py-spy", "record"]
    assert dispatch.removed == ["/tmp/fake-1.json"]


def test_capture_cpu_raises_on_nonzero_exit():
    dispatch = FakeDispatch(profiler_result=ExecResult(137, b"", "killed"))
    cfg = job_pb2.CpuProfile(format=job_pb2.CpuProfile.SPEEDSCOPE)

    with pytest.raises(RuntimeError, match="py-spy record failed"):
        capture_cpu(dispatch, cfg, duration_seconds=5, pid="1")
    # Output temp file is still cleaned up on failure.
    assert dispatch.removed == ["/tmp/fake-1.json"]


def test_capture_threads_returns_stdout_bytes():
    dispatch = FakeDispatch(profiler_result=ExecResult(0, b"Thread 0x1\n  main.py:1", ""))
    data = capture_threads(dispatch, pid="1")
    assert b"Thread 0x1" in data
    assert dispatch.profiler_cmds[0][:2] == ["py-spy", "dump"]


def test_capture_threads_tolerates_non_python_child_with_partial_output():
    """py-spy dump --subprocesses exits non-zero on a non-Python child but still dumps."""
    dispatch = FakeDispatch(
        profiler_result=ExecResult(1, b"Thread 0x1\n  main.py:1", "Failed to find python version from target process")
    )
    data = capture_threads(dispatch, pid="1", subprocesses=True)
    assert b"Thread 0x1" in data


def test_capture_threads_raises_on_real_failure():
    dispatch = FakeDispatch(profiler_result=ExecResult(1, b"", "no such process"))
    with pytest.raises(RuntimeError, match="py-spy dump failed"):
        capture_threads(dispatch, pid="1")


def test_capture_memory_flamegraph_attaches_transforms_reads_file():
    dispatch = FakeDispatch()
    dispatch.files["/tmp/fake-2.html"] = b"<html>flamegraph</html>"  # fake-1 is the .bin trace
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH)

    data = capture_memory_attach(dispatch, cfg, duration_seconds=5, pid="1")

    assert data == b"<html>flamegraph</html>"
    assert dispatch.profiler_cmds[0][:2] == ["memray", "attach"]
    assert dispatch.transform_cmds[0][:2] == ["memray", "flamegraph"]
    assert dispatch.removed == ["/tmp/fake-1.bin", "/tmp/fake-2.html"]


def test_capture_memory_table_returns_transform_stdout():
    dispatch = FakeDispatch(transform_result=ExecResult(0, b"ALLOC SIZE FILE", ""))
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.TABLE)

    data = capture_memory_attach(dispatch, cfg, duration_seconds=5, pid="1")

    assert data == b"ALLOC SIZE FILE"
    assert dispatch.removed == ["/tmp/fake-1.bin"]  # table writes no output file
