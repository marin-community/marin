# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from rigging import telemetry
from rigging.telemetry.probes import nccl, nccl_client, nvidia
from rigging.testing import RecordingTelemetryTransport, nccl_ras_payload


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    telemetry.shutdown(0.01)
    yield
    telemetry.shutdown(0.1)


def _configure(monkeypatch: pytest.MonkeyPatch) -> RecordingTelemetryTransport:
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(
        endpoint="http://finelog.test/v1/telemetry",
        service="probe-test",
        retry_initial=0.001,
        retry_maximum=0.002,
    )
    return transport


def _executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _install_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    nvidia_source: str,
    nccl_source: str,
) -> None:
    _executable(tmp_path / "nvidia-smi", nvidia_source)
    _executable(tmp_path / "nccl-client", nccl_source)
    monkeypatch.setattr(nccl, "_CLIENT_COMMAND", (str(tmp_path / "nccl-client"),))
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")


class _RasConnection:
    def __init__(self, responses: list[bytes]) -> None:
        self._responses = iter(responses)
        self.request = b""

    def __enter__(self) -> "_RasConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def sendall(self, request: bytes) -> None:
        self.request += request

    def shutdown(self, _how: int) -> None:
        pass

    def settimeout(self, _timeout: float) -> None:
        pass

    def recv(self, _size: int) -> bytes:
        return next(self._responses)


def test_nccl_client_queries_documented_json_status(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _RasConnection([b"OK\nOK\n", json.dumps(nccl_ras_payload()).encode(), b""])
    connected_to: list[tuple[tuple[str, int], float]] = []

    def connect(address: tuple[str, int], timeout: float) -> _RasConnection:
        connected_to.append((address, timeout))
        return connection

    monkeypatch.setattr(nccl_client.socket, "create_connection", connect)

    response = nccl_client.query_nccl_ras(address="localhost:28028", timeout=1.2)

    assert connected_to == [(("localhost", 28028), 1.2)]
    assert connection.request == b"TIMEOUT 2\nSET FORMAT json\nVERBOSE STATUS\n"
    assert response == b"OK\nOK\n" + json.dumps(nccl_ras_payload()).encode()


def test_nccl_client_bounds_one_shot_response(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _RasConnection([b"oversized"])
    monkeypatch.setattr(nccl_client.socket, "create_connection", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(nccl_client, "MAX_RESPONSE_BYTES", 8)

    with pytest.raises(nccl_client.ResponseTooLargeError):
        nccl_client.query_nccl_ras(address="localhost:28028", timeout=1)


def test_nvidia_probe_emits_only_stable_hardware_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nvidia_row = (
        "GPU-f81d4fae, 00000000:17:00.0, NVIDIA H100 80GB HBM3, 580.65.06, "
        "81559, Default, Disabled, 700.00, 96.00.68.00.01, 2, 5, 1, Yes, 3, 4, Yes, No"
    )
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source=f"print({nvidia_row!r})",
        nccl_source=f"print({json.dumps(nccl_ras_payload())!r})",
    )
    transport = _configure(monkeypatch)

    session = nvidia.start()
    inventory = transport.record("hardware_inventory", {"gpu_uuid": "GPU-f81d4fae"})
    session.shutdown()

    assert inventory["attributes"]["pci_bus_id"] == "00000000:17:00.0"
    assert inventory["attributes"]["source_temporality"] == telemetry.CURRENT_SNAPSHOT
    assert not any(record["name"] == "communicators" for record in transport.records)


def test_nccl_probe_emits_only_communicator_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source="raise SystemExit(2)",
        nccl_source=f"print({json.dumps(nccl_ras_payload())!r})",
    )
    transport = _configure(monkeypatch)

    session = nccl.start()
    rank = transport.record(
        "communicator_rank_status",
        {"communicator_hash": "0xae94423cfbb2ef4a", "rank": "1"},
    )
    collective = transport.record("collective_operations", {"collective": "AllReduce", "rank": "0"})
    available = transport.record("ras_available", {"outcome": "success"})
    poll_duration = transport.record("ras_poll_duration_seconds", {"outcome": "success"})
    session.shutdown()

    assert rank["attributes"]["rank_state"] == "missing"
    assert rank["attributes"]["unresponsive"] == "true"
    assert rank["attributes"]["rank_host"] == "10.0.0.2"
    assert rank["attributes"]["process_id"] == "5678"
    assert rank["attributes"]["cuda_device"] == "0"
    assert rank["attributes"]["nvml_device"] == "1"
    assert collective["value"] == 12
    assert collective["attributes"]["rank_host"] == "10.0.0.1"
    assert collective["attributes"]["process_id"] == "1234"
    assert collective["attributes"]["cuda_device"] == "0"
    assert collective["attributes"]["nvml_device"] == "3"
    assert collective["attributes"]["source_temporality"] == telemetry.CUMULATIVE_SNAPSHOT
    assert available["value"] == 1
    assert poll_duration["kind"] == "histogram"
    assert not any(record["name"] == "hardware_inventory" for record in transport.records)


def test_nccl_probe_records_unavailable_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source="raise SystemExit(2)",
        nccl_source=f"raise SystemExit({nccl_client.UNAVAILABLE_EXIT_CODE})",
    )
    transport = _configure(monkeypatch)

    session = nccl.start()
    available = transport.record("ras_available", {"outcome": "unavailable"})
    failures = transport.record("ras_poll_failures", {"failure_kind": "unavailable"})
    session.shutdown()

    assert available["value"] == 0
    assert failures["kind"] == "counter"
    assert failures["value"] == 1
    assert failures["attributes"]["source_temporality"] == telemetry.CUMULATIVE_SNAPSHOT


def test_nccl_probe_records_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source="raise SystemExit(2)",
        nccl_source="import threading\nthreading.Event().wait(60)",
    )
    monkeypatch.setattr(nccl, "TIMEOUT", 0.05)
    transport = _configure(monkeypatch)

    session = nccl.start()
    available = transport.record("ras_available", {"outcome": "deadline_exceeded"})
    timeouts = transport.record("ras_poll_timeouts", {})
    session.shutdown()

    assert available["value"] == 0
    assert timeouts["kind"] == "counter"
    assert timeouts["value"] == 1
    assert not any(record["name"] == "communicators" for record in transport.records)


def test_nvidia_probe_falls_back_to_stable_inventory_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = "GPU-f81d4fae, 00000000:17:00.0, NVIDIA H100 80GB HBM3, 580.65.06, 81559"
    nvidia_source = f"""
import sys
if 'compute_mode' in sys.argv[1]:
    raise SystemExit(2)
print({baseline!r})
"""
    _install_commands(monkeypatch, tmp_path, nvidia_source=nvidia_source, nccl_source="raise SystemExit(2)")
    transport = _configure(monkeypatch)

    session = nvidia.start()
    memory = transport.record("gpu_memory_total_bytes", {"gpu_uuid": "GPU-f81d4fae"})
    session.shutdown()

    assert memory["value"] == 81_559 * 1024**2


def test_output_limit_drops_one_probe_without_blocking_its_peer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source="import os\nos.write(1, b'x' * 300_000)",
        nccl_source=f"print({json.dumps(nccl_ras_payload())!r})",
    )
    transport = _configure(monkeypatch)

    nvidia_session = nvidia.start()
    nccl_session = nccl.start()
    transport.record("communicators", {})
    nvidia_session.shutdown()
    nccl_session.shutdown()

    assert not any(record["name"] == "hardware_inventory" for record in transport.records)


def test_shutdown_cancels_and_reaps_active_probe_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nvidia_pid = tmp_path / "nvidia.pid"
    nccl_pid = tmp_path / "nccl.pid"
    blocking_source = """
import os
import pathlib
import threading
pathlib.Path(os.environ['PROBE_PID_PATH']).write_text(str(os.getpid()))
threading.Event().wait(60)
"""
    _executable(tmp_path / "nvidia-smi", blocking_source)
    _executable(tmp_path / "nccl-client", blocking_source)
    monkeypatch.setattr(nccl, "_CLIENT_COMMAND", (str(tmp_path / "nccl-client"),))
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PROBE_PID_PATH", str(nvidia_pid))

    # Give each command a distinct output path while retaining the real process boundary.
    nccl_command = tmp_path / "nccl-client"
    nccl_command.write_text(nccl_command.read_text().replace("PROBE_PID_PATH", "NCCL_PID_PATH"))
    monkeypatch.setenv("NCCL_PID_PATH", str(nccl_pid))

    nvidia_session = nvidia.start()
    nccl_session = nccl.start()
    deadline = time.monotonic() + 2
    while not (nvidia_pid.exists() and nccl_pid.exists()) and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert nvidia_pid.exists() and nccl_pid.exists()

    started = time.monotonic()
    nvidia_session.shutdown(0.5)
    nccl_session.shutdown(0.5)

    assert time.monotonic() - started < 1.2
    for path in (nvidia_pid, nccl_pid):
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)


def test_import_does_not_start_probe_threads() -> None:
    code = """
import threading
before = {thread.ident for thread in threading.enumerate()}
from rigging.telemetry.probes import nccl, nvidia
after = [thread for thread in threading.enumerate() if thread.ident not in before]
assert after == [], after
"""

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3)

    assert result.returncode == 0, result.stderr
