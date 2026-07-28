# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

from iris.cluster.runtime import distributed_diagnostic_probe_v1
from iris.cluster.runtime.distributed_diagnostic_probe_v1 import (
    MAX_BUNDLE_BYTES,
    collect_diagnostic,
    encode_bundle,
)


def _closed_port() -> tuple[socket.socket, int]:
    guard = socket.socket()
    guard.bind(("127.0.0.1", 0))
    return guard, guard.getsockname()[1]


def test_probe_starts_when_strenum_is_unavailable(tmp_path):
    (tmp_path / "sitecustomize.py").write_text("import enum\ndel enum.StrEnum\n")
    environment = os.environ | {"PYTHONPATH": str(tmp_path)}

    result = subprocess.run(
        [sys.executable, str(Path(distributed_diagnostic_probe_v1.__file__)), "--help"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Standalone, bounded distributed GPU diagnostic probe" in result.stdout


def test_collect_diagnostic_unavailable_tools_preserves_process_evidence():
    guard, port = _closed_port()
    try:
        bundle = collect_diagnostic(
            pid=os.getpid(),
            source="/local/job/0",
            attempt_id=3,
            captured_at="2026-07-28T00:00:00",
            timeout=1,
            py_spy="/missing/py-spy",
            ras_port=port,
        )
    finally:
        guard.close()

    assert bundle["source"] == "/local/job/0"
    assert bundle["attempt_id"] == 3
    assert bundle["process"]["status"] == "ok"
    assert f"Pid:\t{os.getpid()}" in bundle["process"]["status_text"]
    assert bundle["process"]["threads"]
    assert {"status_text", "wchan"} <= bundle["process"]["threads"][0].keys()
    assert bundle["runtime"]["target_executable"]["path"]
    assert all(
        {"name", "version", "location", "libraries"} <= package.keys() for package in bundle["runtime"]["packages"]
    )
    assert bundle["threads"]["status"] == "unavailable"
    assert bundle["nccl_ras"]["status"] == "unavailable"
    assert bundle["gpus"]["status"] in {"ok", "unavailable"}
    assert {
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "power.limit",
    } <= set(bundle["gpus"]["fields"])
    assert all(
        name.startswith(("NCCL_", "XLA_", "CUDA_")) or name == "XLA_FLAGS" for name in bundle["environment"]["variables"]
    )
    assert {error["collector"] for error in bundle["errors"]} >= {"threads", "nccl_ras_json"}


def test_collect_diagnostic_reads_documented_nccl_ras_json():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    requests: list[bytes] = []

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            request = b""
            while chunk := connection.recv(4096):
                request += chunk
            requests.append(request)
            connection.sendall(
                b'OK\n{"communicators":[{"hash":"abc","ranks":' b'[{"rank":0,"collective_counts":{"AllReduce":7}}]}]}'
            )

    server = threading.Thread(target=serve)
    server.start()
    try:
        bundle = collect_diagnostic(
            pid=os.getpid(),
            source="/local/job/0",
            attempt_id=0,
            captured_at="2026-07-28T00:00:00",
            timeout=2,
            py_spy="/missing/py-spy",
            ras_port=listener.getsockname()[1],
        )
    finally:
        listener.close()
        server.join(timeout=2)

    assert not server.is_alive()
    assert requests and b"SET FORMAT json\nVERBOSE STATUS\n" in requests[0]
    assert bundle["nccl_ras"]["status"] == "ok"
    assert bundle["nccl_ras"]["response_format"] == "json"
    assert bundle["nccl_ras"]["json"]["report"]["communicators"][0]["hash"] == "abc"
    assert bundle["nccl_ras"]["json"]["raw_response"].startswith("OK\n")


def test_collect_diagnostic_preserves_partial_json_when_ras_falls_back_to_text():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    requests: list[bytes] = []
    responses = [b'OK\n{"communicators":[', b"RAS STATUS PARTIAL\ncommunicator abc"]

    def serve() -> None:
        for response in responses:
            connection, _ = listener.accept()
            with connection:
                request = b""
                while chunk := connection.recv(4096):
                    request += chunk
                requests.append(request)
                connection.sendall(response)

    server = threading.Thread(target=serve)
    server.start()
    try:
        bundle = collect_diagnostic(
            pid=os.getpid(),
            source="/local/job/0",
            attempt_id=0,
            captured_at="2026-07-28T00:00:00",
            timeout=2,
            py_spy="/missing/py-spy",
            ras_port=listener.getsockname()[1],
        )
    finally:
        listener.close()
        server.join(timeout=2)

    assert not server.is_alive()
    assert len(requests) == 2
    assert b"SET FORMAT json" in requests[0]
    assert b"SET FORMAT text" in requests[1]
    assert bundle["nccl_ras"]["status"] == "partial"
    assert bundle["nccl_ras"]["response_format"] == "text"
    assert bundle["nccl_ras"]["json"]["raw_response"] == 'OK\n{"communicators":['
    assert bundle["nccl_ras"]["text"]["raw_response"].startswith("RAS STATUS PARTIAL")


def test_encode_bundle_trims_text_but_preserves_sections():
    bundle = {
        "schema_version": 1,
        "collector_version": "test",
        "captured_at": "2026-07-28T00:00:00",
        "source": "/local/job/0",
        "attempt_id": 0,
        "process": {"status": "ok", "status_text": ""},
        "environment": {"status": "ok", "variables": {}},
        "runtime": {"status": "ok"},
        "nccl_ras": {"status": "ok"},
        "threads": {"status": "ok", "text": "x" * (5 * 1024 * 1024)},
        "gpus": {"status": "ok"},
        "errors": [],
    }

    encoded = encode_bundle(bundle)
    decoded = json.loads(encoded)

    assert len(encoded) <= MAX_BUNDLE_BYTES
    assert decoded["truncated"] is True
    assert {"process", "environment", "runtime", "nccl_ras", "threads", "gpus"} <= decoded.keys()
    assert any(error["collector"] == "bundle" for error in decoded["errors"])
