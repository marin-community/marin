# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from rigging import telemetry
from rigging.telemetry.probes import nccl, nccl_client, nccl_ras, nvidia
from rigging.telemetry.probes.runner import MAX_OUTPUT_BYTES, BoundedCommandRunner, CommandStatus
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


def _nccl_source(payload: dict[str, object]) -> str:
    report = nccl_ras.reduce_response(json.dumps(payload).encode(), detail=nccl_ras.RasDetail.PERIODIC)
    return f"import sys\nsys.stdout.write({nccl_ras.serialize_success(report).decode()!r})"


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


class _RasServer:
    def __init__(self, response: bytes) -> None:
        self._response = response
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def address(self) -> str:
        host, port = self._socket.getsockname()
        return f"{host}:{port}"

    def __enter__(self) -> "_RasServer":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._socket.close()
        self._thread.join(timeout=1)

    def _serve(self) -> None:
        connection, _address = self._socket.accept()
        with connection:
            while connection.recv(64 * 1024):
                pass
            connection.sendall(self._response)


def _large_healthy_nccl_ras_payload() -> dict[str, object]:
    communicators = []
    for communicator_index in range(77):
        size = (8, 64, 128)[communicator_index % 3]
        ranks = [
            {
                "rank": rank,
                "host": f"10.0.{rank // 256}.{rank % 256}",
                "pid": 10_000 + rank,
                "cuda_dev": rank % 4,
                "nvml_dev": rank % 4,
                "status": {
                    "init_state": 0,
                    "async_error": 0,
                    "finalize_called": False,
                    "destroy_flag": False,
                    "abort_flag": False,
                },
                "collective_counts": {
                    "AllGather": 100,
                    "AllReduce": 200,
                    "Broadcast": 0,
                    "Reduce": 0,
                    "ReduceScatter": 100,
                },
            }
            for rank in range(size)
        ]
        communicators.append(
            {
                "hash": f"0x{communicator_index:016x}",
                "secondary_hash": f"0x{communicator_index:016x}:0x{communicator_index + 1:016x}",
                "size": size,
                "ranks_count": size,
                "missing_ranks_count": 0,
                "ranks": ranks,
                "missing_ranks": [],
            }
        )
    return {
        "nccl_version": "2.28.9",
        "cuda_runtime_version": 13000,
        "cuda_driver_version": 13000,
        "communicators_count": len(communicators),
        "communicators": communicators,
        "ras": {"collection_time_sec": 1.25, "timeouts_count": 0},
    }


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

    with pytest.raises(nccl_client.ResponseTooLargeError) as raised:
        nccl_client.query_nccl_ras(address="localhost:28028", timeout=1)

    assert raised.value.observed_bytes == len(b"oversized")
    assert raised.value.limit_bytes == 8


def test_nccl_client_uses_nonverbose_status_for_periodic_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _RasConnection([json.dumps(nccl_ras_payload()).encode(), b""])
    monkeypatch.setattr(nccl_client.socket, "create_connection", lambda *_args, **_kwargs: connection)

    nccl_client.query_nccl_ras(address="localhost:28028", timeout=1, verbose=False)

    assert connection.request == b"TIMEOUT 1\nSET FORMAT json\nSTATUS\n"


def test_nccl_client_reduces_512_rank_response_before_runner_output_limit() -> None:
    raw_response = b"OK\nOK\n" + json.dumps(_large_healthy_nccl_ras_payload()).encode()
    assert len(raw_response) > MAX_OUTPUT_BYTES

    with _RasServer(raw_response) as server:
        result = BoundedCommandRunner().run_result(
            (
                sys.executable,
                "-m",
                nccl_client.__name__,
                "--address",
                server.address,
                "--timeout",
                "3",
                "--detail",
                "periodic",
            ),
            timeout=5,
        )

    assert result.status is CommandStatus.COMPLETED
    assert result.output is not None
    assert result.output.returncode == 0
    assert len(result.output.stdout) < MAX_OUTPUT_BYTES
    client_result = nccl_ras.parse_client_result(result.output.stdout)
    assert isinstance(client_result, nccl_ras.NcclRasSuccess)
    assert client_result.report.input_communicators == 77
    assert client_result.report.invalid_communicators == 0
    assert client_result.report.rank_observations == ()


def test_nccl_client_reports_invalid_communicator_without_dropping_valid_peers() -> None:
    payload = _large_healthy_nccl_ras_payload()
    communicators = cast(list[dict[str, object]], payload["communicators"])
    first = communicators[0]
    first["ranks_count"] = 999
    raw_response = b"OK\nOK\n" + json.dumps(payload).encode()

    report = nccl_ras.reduce_response(raw_response, detail=nccl_ras.RasDetail.PERIODIC)

    assert report.input_communicators == 77
    assert report.emitted_communicators == 76
    assert report.invalid_communicators == 1


def test_nccl_stall_report_retains_only_the_unique_progress_outlier() -> None:
    payload = _large_healthy_nccl_ras_payload()
    communicators = cast(list[dict[str, object]], payload["communicators"])
    payload["communicators"] = communicators[:1]
    payload["communicators_count"] = 1
    communicator = communicators[0]
    ranks = cast(list[dict[str, object]], communicator["ranks"])
    communicator["ranks"] = ranks[:4]
    communicator["size"] = 4
    communicator["ranks_count"] = 4
    outlier = ranks[3]
    counts = cast(dict[str, int], outlier["collective_counts"])
    counts["AllReduce"] = 199
    response = json.dumps(payload).encode()

    periodic = nccl_ras.reduce_response(response, detail=nccl_ras.RasDetail.PERIODIC)
    stall = nccl_ras.reduce_response(response, detail=nccl_ras.RasDetail.STALL)

    assert periodic.rank_observations == ()
    assert len(stall.rank_observations) == 1
    assert stall.rank_observations[0].rank == 3
    assert stall.rank_observations[0].reasons == ("collective_outlier:AllReduce",)


def test_nccl_stall_report_bounds_derived_outlier_reason_by_utf8_bytes() -> None:
    payload = _large_healthy_nccl_ras_payload()
    communicators = cast(list[dict[str, object]], payload["communicators"])
    payload["communicators"] = communicators[:1]
    payload["communicators_count"] = 1
    communicator = communicators[0]
    ranks = cast(list[dict[str, object]], communicator["ranks"])
    communicator["ranks"] = ranks[:4]
    communicator["size"] = 4
    communicator["ranks_count"] = 4
    collective = "界" * 85
    counts = cast(dict[str, int], ranks[3]["collective_counts"])
    counts[collective] = 1

    report = nccl_ras.reduce_response(json.dumps(payload).encode(), detail=nccl_ras.RasDetail.STALL)

    reason = report.rank_observations[0].reasons[0]
    assert reason.startswith("collective_outlier:")
    assert len(reason.encode()) == nccl_ras.MAX_FIELD_BYTES


def test_nccl_reduced_output_preserves_anomalies_before_progress_detail() -> None:
    report = nccl_ras.reduce_response(
        json.dumps(nccl_ras_payload()).encode(),
        detail=nccl_ras.RasDetail.STALL,
    )
    observation = report.rank_observations[0]
    observations = tuple(observation.model_copy(update={"rank": rank}) for rank in range(100))
    progress = tuple(
        nccl_ras.CollectiveProgress(
            communicator_hash=f"0x{index:016x}",
            secondary_hash=f"0x{index:016x}:0x{index + 1:016x}",
            collective=f"Collective{index}",
            minimum=index,
            maximum=index + 1,
        )
        for index in range(nccl_ras.MAX_PROGRESS_SUMMARIES)
    )
    report = report.model_copy(
        update={
            "input_progress_summaries": len(progress),
            "omitted_progress_summaries": 0,
            "input_rank_observations": len(observations),
            "omitted_rank_observations": 0,
            "progress": progress,
            "rank_observations": observations,
        }
    )

    payload = nccl_ras.serialize_success(report)
    parsed = nccl_ras.parse_client_result(payload)

    assert len(payload) <= nccl_ras.MAX_CLIENT_OUTPUT_BYTES
    assert isinstance(parsed, nccl_ras.NcclRasSuccess)
    assert len(parsed.report.rank_observations) == len(observations)
    assert parsed.report.omitted_progress_summaries > 0


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
        nccl_source=_nccl_source(nccl_ras_payload()),
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
        nccl_source=_nccl_source(nccl_ras_payload()),
    )
    transport = _configure(monkeypatch)

    session = nccl.start()
    rank = transport.record(
        "communicator_rank_status",
        {"communicator_hash": "0xae94423cfbb2ef4a", "rank": "1"},
    )
    collective = transport.record(
        "collective_operations",
        {"collective": "AllReduce", "rank_statistic": "minimum"},
    )
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
    assert collective["attributes"]["source_temporality"] == telemetry.CUMULATIVE_SNAPSHOT
    assert available["value"] == 1
    assert poll_duration["kind"] == "histogram"
    assert not any(record["name"] == "hardware_inventory" for record in transport.records)


def test_nvidia_probe_summarizes_healthy_devices_without_zero_error_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nvidia_row = (
        "GPU-f81d4fae, 00000000:17:00.0, NVIDIA H100 80GB HBM3, 580.65.06, "
        "81559, Default, Disabled, 700.00, 96.00.68.00.01, 0, 0, 0, No, 0, 0, No, No"
    )
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source=f"print({nvidia_row!r})",
        nccl_source="raise SystemExit(2)",
    )
    transport = _configure(monkeypatch)

    session = nvidia.start()
    available = transport.record("nvidia_health_available", {"outcome": "success"})
    healthy = transport.record("gpu_devices", {"device_state": "healthy"})
    session.shutdown()

    assert available["value"] == 1
    assert healthy["value"] == 1
    assert not any(
        record["name"]
        in {
            "gpu_ecc_uncorrected_errors",
            "gpu_retired_pages",
            "gpu_retired_pages_pending",
            "gpu_row_remapped_rows",
            "gpu_row_remap_pending",
            "gpu_row_remap_failures",
        }
        for record in transport.records
    )


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
    available = transport.record("nvidia_health_available", {"outcome": "success_baseline"})
    session.shutdown()

    assert memory["value"] == 81_559 * 1024**2
    assert available["value"] == 1


def test_output_limit_drops_one_probe_without_blocking_its_peer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source="import os\nos.write(1, b'x' * 300_000)",
        nccl_source=_nccl_source(nccl_ras_payload()),
    )
    transport = _configure(monkeypatch)

    nvidia_session = nvidia.start()
    nccl_session = nccl.start()
    transport.record("communicators", {})
    nvidia_session.shutdown()
    nccl_session.shutdown()

    assert not any(record["name"] == "hardware_inventory" for record in transport.records)


def test_nccl_parent_output_limit_records_which_boundary_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_commands(
        monkeypatch,
        tmp_path,
        nvidia_source="raise SystemExit(2)",
        nccl_source="import os\nos.write(1, b'x' * 300_000)",
    )
    transport = _configure(monkeypatch)

    session = nccl.start()
    failure = transport.record("ras_poll_failures", {"failure_kind": "runner_output_limit"})
    session.shutdown()

    assert failure["attributes"]["observed_bytes"] == str(MAX_OUTPUT_BYTES + 1)
    assert failure["attributes"]["limit_bytes"] == str(MAX_OUTPUT_BYTES)


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
