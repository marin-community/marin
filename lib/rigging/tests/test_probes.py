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
from rigging.telemetry.probes import nccl, nvidia
from rigging.testing import RecordingTelemetryTransport


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
    _executable(tmp_path / "ncclras", nccl_source)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")


def _nccl_payload() -> dict[str, object]:
    return {
        "nccl_version": "2.29.1",
        "cuda_runtime_version": 13000,
        "cuda_driver_version": 13000,
        "communicators_count": 1,
        "communicators": [
            {
                "hash": "0xae94423cfbb2ef4a",
                "secondary_hash": "0xb7e7187447156001:0xb8242ed28a71381e",
                "size": 2,
                "ranks_count": 1,
                "missing_ranks_count": 1,
                "ranks": [
                    {
                        "rank": 0,
                        "status": {
                            "init_state": 0,
                            "async_error": 0,
                            "finalize_called": False,
                            "destroy_flag": False,
                            "abort_flag": False,
                        },
                        "collective_counts": {"AllReduce": 12},
                    }
                ],
                "missing_ranks": [
                    {
                        "rank": 1,
                        "status": {"unresponsive": True, "considered_dead": False},
                    }
                ],
            }
        ],
        "ras": {"collection_time_sec": 0.125, "timeouts_count": 2},
    }


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
        nccl_source=f"print({json.dumps(_nccl_payload())!r})",
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
        nccl_source=f"print({json.dumps(_nccl_payload())!r})",
    )
    transport = _configure(monkeypatch)

    session = nccl.start()
    rank = transport.record(
        "communicator_rank_status",
        {"communicator_hash": "0xae94423cfbb2ef4a", "rank": "1"},
    )
    collective = transport.record("collective_operations", {"collective": "AllReduce", "rank": "0"})
    session.shutdown()

    assert rank["attributes"]["rank_state"] == "missing"
    assert rank["attributes"]["unresponsive"] == "true"
    assert collective["value"] == 12
    assert collective["attributes"]["source_temporality"] == telemetry.CUMULATIVE_SNAPSHOT
    assert not any(record["name"] == "hardware_inventory" for record in transport.records)


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
        nccl_source=f"print({json.dumps(_nccl_payload())!r})",
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
    _executable(tmp_path / "ncclras", blocking_source)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PROBE_PID_PATH", str(nvidia_pid))

    # Give each command a distinct output path while retaining the real process boundary.
    nccl_command = tmp_path / "ncclras"
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
