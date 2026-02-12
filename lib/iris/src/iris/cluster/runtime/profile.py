# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared profiling command construction for CPU (py-spy) and memory (memray).

Pure functions — no I/O, no subprocess calls. Both the Docker and process
runtimes build their profiler commands through this module, eliminating
duplicated format maps and fragile index-based command construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from iris.rpc import cluster_pb2

CPU_FORMAT_MAP: dict[int, tuple[str, str]] = {
    cluster_pb2.CpuProfile.FLAMEGRAPH: ("flamegraph", "svg"),
    cluster_pb2.CpuProfile.SPEEDSCOPE: ("speedscope", "json"),
    cluster_pb2.CpuProfile.RAW: ("raw", "txt"),
}

MEMORY_FORMAT_MAP: dict[int, tuple[str, str]] = {
    cluster_pb2.MemoryProfile.FLAMEGRAPH: ("flamegraph", "html"),
    cluster_pb2.MemoryProfile.TABLE: ("table", "txt"),
    cluster_pb2.MemoryProfile.STATS: ("stats", "json"),
}


@dataclass(frozen=True)
class CpuProfileSpec:
    py_spy_format: str
    ext: str
    rate_hz: int
    duration_seconds: int
    pid: str


@dataclass(frozen=True)
class MemoryProfileSpec:
    reporter: str  # "flamegraph", "table", "stats"
    ext: str
    duration_seconds: int
    pid: str
    leaks: bool

    @property
    def output_is_file(self) -> bool:
        """Flamegraph writes to a file; table/stats write to stdout."""
        return self.reporter == "flamegraph"


def resolve_cpu_spec(cpu_config: cluster_pb2.CpuProfile, duration_seconds: int, pid: str) -> CpuProfileSpec:
    py_spy_format, ext = CPU_FORMAT_MAP.get(cpu_config.format, ("flamegraph", "svg"))
    rate_hz = cpu_config.rate_hz if cpu_config.rate_hz > 0 else 100
    return CpuProfileSpec(
        py_spy_format=py_spy_format,
        ext=ext,
        rate_hz=rate_hz,
        duration_seconds=duration_seconds,
        pid=pid,
    )


def resolve_memory_spec(memory_config: cluster_pb2.MemoryProfile, duration_seconds: int, pid: str) -> MemoryProfileSpec:
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
    ]


def build_memray_attach_cmd(spec: MemoryProfileSpec, memray_bin: str, trace_path: str) -> list[str]:
    cmd = [memray_bin, "attach", spec.pid, "--duration", str(spec.duration_seconds), "--output", trace_path]
    if spec.leaks:
        cmd.append("--aggregate")
    return cmd


def build_memray_transform_cmd(spec: MemoryProfileSpec, memray_bin: str, trace_path: str, output_path: str) -> list[str]:
    """Build the memray transform command.

    For flamegraph, writes to output_path. For table/stats, output goes to stdout.
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
        return [memray_bin, "stats", "--json", trace_path]
    else:
        raise RuntimeError(f"Unknown memray reporter: {spec.reporter}")
