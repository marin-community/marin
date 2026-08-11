# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for profile command construction helpers.

Focuses on format-to-flag mapping, default handling, and CLI structure —
not on pass-through of constructor arguments.
"""

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest
from iris.cluster.runtime.profile import (
    ExecResult,
    _run_memray_profile,
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_profile_row,
    build_pyspy_cmd,
    build_pyspy_dump_cmd,
    capture_cpu,
    capture_memory_attach,
    capture_threads,
    resolve_cpu_spec,
    resolve_memory_spec,
)
from iris.rpc import job_pb2

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
