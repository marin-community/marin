# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for profile command construction helpers."""


from iris.cluster.runtime.profile import (
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    resolve_cpu_spec,
    resolve_memory_spec,
)
from iris.rpc import cluster_pb2


class TestResolveCpuSpec:
    def test_flamegraph_format(self):
        cfg = cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH, rate_hz=200)
        spec = resolve_cpu_spec(cfg, duration_seconds=10, pid="42")

        assert spec.py_spy_format == "flamegraph"
        assert spec.ext == "svg"
        assert spec.rate_hz == 200
        assert spec.duration_seconds == 10
        assert spec.pid == "42"

    def test_speedscope_format(self):
        cfg = cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.SPEEDSCOPE)
        spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")

        assert spec.py_spy_format == "speedscope"
        assert spec.ext == "json"

    def test_default_rate_hz(self):
        cfg = cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH, rate_hz=0)
        spec = resolve_cpu_spec(cfg, duration_seconds=5, pid="1")

        assert spec.rate_hz == 100


class TestResolveMemorySpec:
    def test_flamegraph_is_file_output(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH, leaks=True)
        spec = resolve_memory_spec(cfg, duration_seconds=10, pid="1")

        assert spec.reporter == "flamegraph"
        assert spec.output_is_file is True
        assert spec.leaks is True

    def test_table_is_stdout_output(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.TABLE)
        spec = resolve_memory_spec(cfg, duration_seconds=5, pid="99")

        assert spec.reporter == "table"
        assert spec.output_is_file is False

    def test_stats_format(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.STATS)
        spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")

        assert spec.reporter == "stats"
        assert spec.ext == "json"


class TestBuildPyspyCmd:
    def test_contains_required_flags(self):
        cfg = cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH, rate_hz=150)
        spec = resolve_cpu_spec(cfg, duration_seconds=10, pid="42")
        cmd = build_pyspy_cmd(spec, py_spy_bin="/usr/bin/py-spy", output_path="/tmp/out.svg")

        assert cmd[0] == "/usr/bin/py-spy"
        assert "record" in cmd
        assert "--pid" in cmd
        assert cmd[cmd.index("--pid") + 1] == "42"
        assert "--duration" in cmd
        assert cmd[cmd.index("--duration") + 1] == "10"
        assert "--rate" in cmd
        assert cmd[cmd.index("--rate") + 1] == "150"
        assert "--output" in cmd
        assert cmd[cmd.index("--output") + 1] == "/tmp/out.svg"
        assert "--subprocesses" in cmd


class TestBuildMemrayCmd:
    def test_attach_cmd_with_leaks(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH, leaks=True)
        spec = resolve_memory_spec(cfg, duration_seconds=10, pid="5")
        cmd = build_memray_attach_cmd(spec, memray_bin="memray", trace_path="/tmp/trace.bin")

        assert cmd[0] == "memray"
        assert "attach" in cmd
        assert "5" in cmd
        assert "--aggregate" in cmd

    def test_attach_cmd_without_leaks(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.TABLE, leaks=False)
        spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
        cmd = build_memray_attach_cmd(spec, memray_bin="memray", trace_path="/tmp/trace.bin")

        assert "--aggregate" not in cmd

    def test_transform_flamegraph_writes_to_file(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH, leaks=True)
        spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
        cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="/tmp/o.html")

        assert "flamegraph" in cmd
        assert "--leaks" in cmd
        assert "--output" in cmd
        assert "/tmp/o.html" in cmd

    def test_transform_table_uses_stdout(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.TABLE)
        spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
        cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="")

        assert "table" in cmd
        assert "--output" not in cmd

    def test_transform_stats_includes_json_flag(self):
        cfg = cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.STATS)
        spec = resolve_memory_spec(cfg, duration_seconds=5, pid="1")
        cmd = build_memray_transform_cmd(spec, memray_bin="memray", trace_path="/tmp/t.bin", output_path="")

        assert "stats" in cmd
        assert "--json" in cmd
