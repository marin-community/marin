# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for profile command construction helpers.

Focuses on format-to-flag mapping, default handling, and CLI structure —
not on pass-through of constructor arguments.
"""

import json
import os

import pytest
from iris.cluster.runtime.profile import (
    _run_memray_profile,
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    resolve_cpu_spec,
    resolve_memory_spec,
    run_pyspy_dump,
)
from iris.rpc import job_pb2

try:
    import memray
except ImportError:
    memray = None
needs_memray = pytest.mark.skipif(memray is None, reason="memray not installed")

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
# run_pyspy_dump: workaround for https://github.com/benfred/py-spy/issues/846
# (py-spy --subprocesses exits non-zero when it walks into a non-Python child)
# ---------------------------------------------------------------------------


def _stub_subprocess_run(monkeypatch, *, returncode: int, stdout: str, stderr: str):
    from subprocess import CompletedProcess

    def fake_run(cmd, **kwargs):
        return CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr("iris.cluster.runtime.profile.subprocess.run", fake_run)


def test_run_pyspy_dump_returns_partial_stdout_on_non_python_child(monkeypatch):
    _stub_subprocess_run(
        monkeypatch,
        returncode=1,
        stdout="Process 1: python -u main.py\nThread 1 (idle): MainThread\n",
        stderr="Error: Failed to find python version from target process\n",
    )
    out = run_pyspy_dump("1", subprocesses=True)
    assert b"Process 1: python -u main.py" in out


def test_run_pyspy_dump_raises_when_real_failure(monkeypatch):
    _stub_subprocess_run(
        monkeypatch,
        returncode=1,
        stdout="",
        stderr="Error: Permission denied (ptrace)\n",
    )
    with pytest.raises(RuntimeError, match="Permission denied"):
        run_pyspy_dump("1", subprocesses=True)


def test_run_pyspy_dump_raises_when_stdout_empty_even_with_known_error(monkeypatch):
    """If we never got the parent dump, propagate the failure — empty output is not partial success."""
    _stub_subprocess_run(
        monkeypatch,
        returncode=1,
        stdout="",
        stderr="Error: Failed to find python version from target process\n",
    )
    with pytest.raises(RuntimeError):
        run_pyspy_dump("1", subprocesses=True)


def test_run_pyspy_dump_raises_when_subprocesses_disabled(monkeypatch):
    """Without --subprocesses, py-spy can't have walked into a child — treat any failure as fatal."""
    _stub_subprocess_run(
        monkeypatch,
        returncode=1,
        stdout="Process 1: python -u main.py\n",
        stderr="Error: Failed to find python version from target process\n",
    )
    with pytest.raises(RuntimeError):
        run_pyspy_dump("1", subprocesses=False)


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
    import threading

    results: list = []

    def _alloc():
        for _ in range(duration_seconds * 100):
            results.append(bytearray(1024))
            import time

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


@needs_memray
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


@needs_memray
def test_run_memray_profile_stats_returns_valid_json():
    """Stats reporter returns parseable JSON, not a file-path string."""
    _allocate_during(1)
    cfg = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.STATS, leaks=False)
    pid = str(os.getpid())
    result = _run_memray_profile(pid, duration_seconds=1, memory_config=cfg)
    data = json.loads(result)
    assert "total_num_allocations" in data
