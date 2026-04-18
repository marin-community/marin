# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for profile command construction helpers.

Focuses on format-to-flag mapping, default handling, and CLI structure —
not on pass-through of constructor arguments.
"""

import pytest

from iris.cluster.runtime.profile import (
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    resolve_cpu_spec,
    resolve_memory_spec,
)
from iris.rpc import cluster_pb2

# ---------------------------------------------------------------------------
# resolve_cpu_spec: enum → (py_spy_format, ext) mapping and defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proto_format, expected_format, expected_ext",
    [
        (cluster_pb2.CpuProfile.FLAMEGRAPH, "flamegraph", "svg"),
        (cluster_pb2.CpuProfile.SPEEDSCOPE, "speedscope", "json"),
        (cluster_pb2.CpuProfile.RAW, "raw", "txt"),
    ],
)
def test_resolve_cpu_spec_maps_format_to_pyspy_format_and_extension(proto_format, expected_format, expected_ext):
    cfg = cluster_pb2.CpuProfile(format=proto_format, rate_hz=100)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    assert spec.py_spy_format == expected_format
    assert spec.ext == expected_ext


def test_resolve_cpu_spec_defaults_rate_hz_when_zero():
    cfg = cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH, rate_hz=0)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    assert spec.rate_hz == 20


def test_resolve_cpu_spec_preserves_nonzero_rate_hz():
    cfg = cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH, rate_hz=250)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    assert spec.rate_hz == 250


# ---------------------------------------------------------------------------
# resolve_memory_spec: enum → (reporter, ext) mapping and output_is_file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proto_format, expected_reporter, expected_ext, expected_is_file",
    [
        (cluster_pb2.MemoryProfile.FLAMEGRAPH, "flamegraph", "html", True),
        (cluster_pb2.MemoryProfile.TABLE, "table", "txt", False),
        (cluster_pb2.MemoryProfile.STATS, "stats", "json", False),
    ],
)
def test_resolve_memory_spec_maps_format(proto_format, expected_reporter, expected_ext, expected_is_file):
    cfg = cluster_pb2.MemoryProfile(format=proto_format)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    assert spec.reporter == expected_reporter
    assert spec.ext == expected_ext
    assert spec.output_is_file is expected_is_file


# ---------------------------------------------------------------------------
# build_pyspy_cmd: verify CLI flag structure
# ---------------------------------------------------------------------------


def test_build_pyspy_cmd_includes_subprocesses_flag_by_default():
    cfg = cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH, rate_hz=100)
    spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_pyspy_cmd(spec, py_spy_bin="py-spy", output_path="/tmp/out.svg")
    assert "--subprocesses" in cmd


# ---------------------------------------------------------------------------
# build_memray_attach_cmd: --aggregate flag depends on leaks
# ---------------------------------------------------------------------------


def test_memray_attach_includes_aggregate_when_leaks_enabled():
    cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH, leaks=True)
    spec = resolve_memory_spec(cfg, duration_seconds=10, pid="5")
    cmd = build_memray_attach_cmd(spec, memray_bin="memray", trace_path="/tmp/trace.bin")
    assert "--aggregate" in cmd


def test_memray_attach_excludes_aggregate_when_leaks_disabled():
    cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.TABLE, leaks=False)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_attach_cmd(spec, memray_bin="memray", trace_path="/tmp/trace.bin")
    assert "--aggregate" not in cmd


# ---------------------------------------------------------------------------
# build_memray_transform_cmd: reporter determines subcommand and output mode
# ---------------------------------------------------------------------------


def test_memray_transform_flamegraph_writes_to_file_with_leaks():
    cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH, leaks=True)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="/tmp/o.html")

    assert "flamegraph" in cmd
    assert "--leaks" in cmd
    assert "--output" in cmd
    assert "/tmp/o.html" in cmd


def test_memray_transform_table_does_not_write_to_file():
    cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.TABLE)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="")

    assert "--output" not in cmd


def test_memray_transform_stats_includes_json_flag():
    cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.STATS)
    spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
    cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="")

    assert "stats" in cmd
    assert "--json" in cmd
