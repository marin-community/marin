# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the swe_rebench_trace tracer payload.

The tracer module is bind-mounted into untrusted containers and loaded via
PYTHONSTARTUP. We can't run it inside a real sandbox here, so these tests
exercise the framing and the filtering by reloading the module with a
controlled MARIN_TRACE_FD pointing at a pipe.
"""

from __future__ import annotations

import importlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

from experiments.swe_rebench_trace.run_one import _iter_trace_records


def _read_one_record(buf: bytes) -> dict | None:
    if len(buf) < 4:
        return None
    (length,) = struct.unpack(">I", buf[:4])
    if len(buf) < 4 + length:
        return None
    return json.loads(buf[4 : 4 + length].decode("utf-8"))


@pytest.fixture
def fresh_tracer(monkeypatch, tmp_path: Path):
    """Reload the tracer module with the trace fd pointing at a temp file."""
    trace_path = tmp_path / "trace.bin"
    trace_fd = os.open(trace_path, os.O_WRONLY | os.O_CREAT, 0o644)

    monkeypatch.setenv("MARIN_TRACE_FD", str(trace_fd))
    monkeypatch.setenv("MARIN_TRACE_ROOTS", str(tmp_path))
    monkeypatch.setenv("MARIN_TRACE_MAX_EVENTS", "1000")

    sys.modules.pop("experiments.swe_rebench_trace.tracer", None)
    tracer = importlib.import_module("experiments.swe_rebench_trace.tracer")

    yield tracer, trace_path

    tracer._disable_tracer()
    try:
        os.close(trace_fd)
    except OSError:
        pass


def test_emit_writes_framed_json_record(fresh_tracer, tmp_path: Path):
    tracer, trace_path = fresh_tracer
    tracer._emit({"e": "call", "f": str(tmp_path / "x.py"), "l": 1, "n": "main"})
    # Force a flush by closing the underlying fd handle in the test.
    os.fsync(tracer._TRACE_FD)
    data = trace_path.read_bytes()
    record = _read_one_record(data)
    assert record == {"e": "call", "f": str(tmp_path / "x.py"), "l": 1, "n": "main"}


def test_path_filter_keeps_in_root(fresh_tracer, tmp_path: Path):
    tracer, _ = fresh_tracer
    in_root = str(tmp_path / "src" / "module.py")
    out_of_root = "/usr/lib/python3.11/json/decoder.py"
    assert tracer._path_in_roots(in_root)
    assert not tracer._path_in_roots(out_of_root)


def test_truncation_caps_emit(fresh_tracer, tmp_path: Path):
    tracer, trace_path = fresh_tracer
    # Emit 1500 events; cap is 1000.
    for i in range(1500):
        tracer._emit({"e": "call", "f": str(tmp_path / "x.py"), "l": i, "n": "f"})
    os.fsync(tracer._TRACE_FD)
    assert tracer._TRUNCATED is True
    # We should have written exactly 1000 records.
    events = list(_iter_trace_records(iter([trace_path.read_bytes()])))
    assert len(events) == 1000


def test_iter_trace_records_decodes_partial_stream():
    """The framed-record decoder used by run_one parses tracer.py output."""
    payloads = [
        json.dumps({"e": "call", "f": "/x", "l": 1, "n": "a"}).encode(),
        json.dumps({"e": "return", "f": "/x", "l": 2, "n": "a"}).encode(),
    ]
    framed = b"".join(struct.pack(">I", len(p)) + p for p in payloads)
    # Hand it the bytes split across two chunks to exercise the buffering.
    chunks = [framed[: len(framed) // 2], framed[len(framed) // 2 :]]
    events = list(_iter_trace_records(iter(chunks)))
    assert events == [
        {"e": "call", "f": "/x", "l": 1, "n": "a"},
        {"e": "return", "f": "/x", "l": 2, "n": "a"},
    ]


def test_install_is_noop_without_env(monkeypatch):
    """Without MARIN_TRACE_FD the install() call must do nothing (safe to import)."""
    monkeypatch.delenv("MARIN_TRACE_FD", raising=False)
    sys.modules.pop("experiments.swe_rebench_trace.tracer", None)
    tracer = importlib.import_module("experiments.swe_rebench_trace.tracer")
    assert tracer._TRACE_ENABLED is False
